import { appendFile, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { spawn, execFile } from 'node:child_process'
import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
import os from 'node:os'
import { promisify } from 'node:util'
import dotenv from 'dotenv'
import express from 'express'
import YAML from 'yaml'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
dotenv.config({ path: path.join(ROOT, '.env') })
const config = YAML.parse(await readFile(path.join(ROOT, 'config.yaml'), 'utf8'))
const runFile = promisify(execFile)
const jobsRoot = path.resolve(ROOT, process.env.JOBS_DIR || config.paths.jobs)
const outputsRoot = path.resolve(ROOT, process.env.OUTPUT_DIR || config.paths.outputs)
const logsRoot = path.join(ROOT, 'logs')
const python = process.env.PYTHON_EXECUTABLE || path.join(ROOT, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python')
const queue = []
const events = new Map()
const active = new Map()
const pendingUpdates = new Map()
const jobWriteLocks = new Map()
let queueRunning = false
let doctorCache = { status: 'checking', checks: {} }

const supportedModes = new Set(['video', 'shorts'])
const supportedLanguages = new Set(['English', 'Hindi'])
const supportedStyles = new Set(['cinematic', 'realistic', 'storybook', 'cartoon', 'anime'])
const supportedMoods = new Set(['auto', 'calm', 'epic', 'happy', 'sad', 'mystery'])
const supportedEffects = new Set(['minimal', 'cinematic', 'dynamic'])
const supportedQualities = new Set(['fast', 'balanced', 'high'])

function emit(jobId, event, data) {
  for (const response of events.get(jobId) || []) {
    response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
  }
}

function safeJobId(value) {
  return /^[0-9a-f-]{36}$/i.test(value)
}

function jobPath(id) {
  if (!safeJobId(id)) throw new Error('Invalid job ID.')
  return path.join(jobsRoot, id, 'job.json')
}

async function retryPermissionError(operation) {
  let delay = 200
  for (let attempt = 0; attempt < 10; attempt += 1) {
    try {
      return await operation()
    } catch (error) {
      if (!['EPERM', 'EACCES'].includes(error.code) || attempt === 9) throw error
      await new Promise((resolve) => setTimeout(resolve, delay))
      delay += 200
    }
  }
}

async function persist(job) {
  const previous = jobWriteLocks.get(job.id) || Promise.resolve()
  const current = previous.catch(() => {}).then(async () => {
    const target = jobPath(job.id)
    const temporary = `${target}.${process.pid}.${randomUUID()}.part`
    const contents = `${JSON.stringify(job, null, 2)}\n`
    try {
      await mkdir(path.dirname(target), { recursive: true })
      await retryPermissionError(async () => {
        await writeFile(temporary, contents, 'utf8')
        await rename(temporary, target)
      })
    } catch (error) {
      console.error(`[job ${job.id}] Atomic job-state save failed; falling back to direct write: ${error.message}`)
      try {
        await writeFile(target, contents, 'utf8')
      } catch (fallbackError) {
        console.error(`[job ${job.id}] Job-state save failed; continuing without a progress snapshot: ${fallbackError.message}`)
      }
    }
    emit(job.id, 'job', job)
    return job
  })
  jobWriteLocks.set(job.id, current)
  try {
    return await current
  } finally {
    if (jobWriteLocks.get(job.id) === current) jobWriteLocks.delete(job.id)
  }
}

async function getJob(id) {
  try {
    const contents = await retryPermissionError(() => readFile(jobPath(id), 'utf8'))
    return JSON.parse(contents)
  } catch (error) {
    if (error.code === 'ENOENT') return null
    throw error
  }
}

async function listJobs() {
  await mkdir(jobsRoot, { recursive: true })
  const directories = await readdir(jobsRoot, { withFileTypes: true })
  const jobs = await Promise.all(directories.filter((item) => item.isDirectory()).map((item) => getJob(item.name)))
  return jobs.filter(Boolean).sort((left, right) => right.created_at - left.created_at)
}

async function logLine(jobId, line) {
  await mkdir(logsRoot, { recursive: true })
  await appendFile(path.join(logsRoot, `job-${jobId}.log`), `${new Date().toISOString()} ${line}\n`, 'utf8')
  console.info(`[job ${jobId}] ${line}`)
}

async function updateFromWorker(jobId, message) {
  const current = await getJob(jobId)
  if (!current || current.status === 'canceled') return
  if (message.event === 'log') {
    await logLine(jobId, JSON.stringify(message))
    return
  }
  if (message.event === 'stage') {
    current.stage = message.stage
    current.stage_status = message.status
    current.progress = message.progress ?? 0
    current.total = message.total ?? 1
    current.stage_started_at = Date.now() / 1000
    current.stage_results = {
      ...current.stage_results,
      [message.stage]: { status: message.status, detail: message.detail ?? null },
    }
  } else if (message.event === 'completed') {
    Object.assign(current, message.result)
  } else if (message.event === 'failed') {
    current.status = 'failed'
    current.stage = message.stage || current.stage
    current.stage_status = 'failed'
    current.error = message.error
    current.stage_results = {
      ...current.stage_results,
      [current.stage]: { status: 'failed', detail: message.error },
    }
  } else if (message.event === 'needs_review') {
    current.status = 'needs_review'
    current.stage = message.stage || current.stage
    current.stage_status = 'needs_review'
    current.error = message.error
    current.qa_report = message.qa_report || current.qa_report
  } else if (message.event === 'worker_error') {
    await logLine(jobId, message.error)
    return
  }
  current.updated_at = Date.now() / 1000
  await persist(current)
}

function serializeWorkerUpdate(jobId, message) {
  const previous = pendingUpdates.get(jobId) || Promise.resolve()
  const next = previous.then(() => updateFromWorker(jobId, message)).catch((error) => logLine(jobId, `Worker progress update failed: ${error.message}`))
  pendingUpdates.set(jobId, next)
  return next
}

async function startJob(job) {
  queueRunning = true
  const current = await getJob(job.id)
  if (!current || current.status !== 'queued') {
    queueRunning = false
    runQueue()
    return
  }
  let preflight
  try {
    const encoded = Buffer.from(JSON.stringify(current), 'utf8').toString('base64')
    const { stdout } = await runFile(python, ['-m', 'workers.doctor', '--preflight-json', encoded], {
      cwd: ROOT,
      timeout: 30000,
      windowsHide: true,
    })
    preflight = JSON.parse(stdout)
  } catch (error) {
    try {
      preflight = JSON.parse(error.stdout || '')
    } catch {
      preflight = {
        status: 'failed',
        checks: { runtime: { status: 'failed', detail: `Preflight could not run: ${error.message}` } },
      }
    }
  }
  if (preflight.status !== 'passed') {
    const failures = Object.entries(preflight.checks || {})
      .filter(([, result]) => result.status !== 'passed')
      .map(([name, result]) => `${name}: ${result.detail}`)
    const reason = failures.length ? failures.join('; ') : 'Preflight did not pass.'
    current.status = 'failed'
    current.stage = 'preflight'
    current.stage_status = 'failed'
    current.error = `Preflight failed: ${reason}`
    current.qa_report = {
      status: 'failed',
      checks: { preflight: { status: 'FAIL', reason, numbers: preflight.checks || {} } },
      metrics: {},
    }
    current.updated_at = Date.now() / 1000
    await mkdir(path.join(jobsRoot, current.id), { recursive: true })
    await writeFile(path.join(jobsRoot, current.id, 'qa_report.json'), `${JSON.stringify(current.qa_report, null, 2)}\n`, 'utf8')
    await persist(current)
    await logLine(current.id, current.error)
    queueRunning = false
    runQueue()
    return
  }
  current.preflight_checks = preflight.checks
  current.status = 'running'
  current.updated_at = Date.now() / 1000
  await persist(current)
  const child = spawn(python, ['-u', '-m', 'workers.pipeline', jobPath(job.id)], {
    cwd: ROOT,
    env: { ...process.env, PYTHONUTF8: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    detached: process.platform !== 'win32',
  })
  active.set(job.id, child)
  const logPath = path.join(logsRoot, `job-${job.id}.log`)
  await mkdir(logsRoot, { recursive: true })
  let stdoutBuffer = ''
  child.stdout.setEncoding('utf8')
  child.stdout.on('data', (chunk) => {
    stdoutBuffer += chunk
    const lines = stdoutBuffer.split(/\r?\n/)
    stdoutBuffer = lines.pop() || ''
    for (const line of lines.filter(Boolean)) {
      try {
        void serializeWorkerUpdate(job.id, JSON.parse(line))
      } catch {
        void logLine(job.id, line)
      }
    }
  })
  child.stderr.setEncoding('utf8')
  child.stderr.on('data', (chunk) => void logLine(job.id, chunk.trimEnd()))
  child.on('error', async (error) => {
    await logLine(job.id, `Could not start Python worker: ${error.message}`)
    const failed = await getJob(job.id)
    if (failed) {
      failed.status = 'failed'
      failed.stage_status = 'failed'
      failed.error = `Could not start local Python worker (${python}): ${error.message}`
      await persist(failed)
    }
  })
  child.on('close', async (code) => {
    if (stdoutBuffer.trim()) {
      try {
        await serializeWorkerUpdate(job.id, JSON.parse(stdoutBuffer))
      } catch {
        await logLine(job.id, stdoutBuffer)
      }
    }
    await pendingUpdates.get(job.id)
    pendingUpdates.delete(job.id)
    active.delete(job.id)
    const finished = await getJob(job.id)
    if (finished?.status === 'running') {
      finished.status = code === 0 ? 'failed' : 'failed'
      finished.stage_status = 'failed'
      finished.error = `Local pipeline exited with code ${code} before reporting completion. See ${logPath}.`
      await persist(finished)
    }
    queueRunning = false
    runQueue()
  })
}

function runQueue() {
  if (queueRunning) return
  const next = queue.shift()
  if (next) void startJob(next)
}

function validateJob(body) {
  const mode = body.mode
  const duration = Number(body.duration)
  const min = mode === 'shorts' ? 15 : 60
  const max = mode === 'shorts' ? 60 : 900
  if (!supportedModes.has(mode)) return 'Choose Shorts or Video.'
  if (!Number.isInteger(duration) || duration < min || duration > max) return `Duration must be ${min}–${max} seconds.`
  if (typeof body.prompt !== 'string' || body.prompt.trim().length < 8 || body.prompt.length > 3000) return 'Prompt must be 8–3000 characters.'
  if (!supportedLanguages.has(body.language)) return 'Choose English or Hindi.'
  if (!supportedStyles.has(body.style)) return 'Choose a supported visual style.'
  if (!['male', 'female'].includes(body.voice)) return 'Choose a male or female narrator.'
  if (body.music_mood && !supportedMoods.has(body.music_mood)) return 'Choose a supported music mood.'
  if (body.effects_level && !supportedEffects.has(body.effects_level)) return 'Choose a supported effects level.'
  if (body.quality && !supportedQualities.has(body.quality)) return 'Choose Fast, Balanced or High quality.'
  if (body.scene_count !== undefined && (!Number.isInteger(body.scene_count) || body.scene_count < 2 || body.scene_count > 30)) return 'Scene count must be an integer from 2 to 30.'
  return null
}

async function doctor() {
  try {
    const { stdout } = await runFile(python, ['-m', 'workers.doctor', '--json'], { cwd: ROOT, timeout: 30000, windowsHide: true })
    doctorCache = JSON.parse(stdout)
  } catch (error) {
    const output = error.stdout || ''
    try {
      doctorCache = JSON.parse(output)
    } catch {
      doctorCache = { status: 'error', checks: { python: { status: 'failed', detail: error.message } } }
    }
  }
  return doctorCache
}

export function createApp() {
  const app = express()
  app.disable('x-powered-by')
  app.use(express.json({ limit: '64kb' }))
  app.get('/api/health', (_request, response) => response.json(doctorCache))
  app.get('/api/jobs', async (_request, response, next) => {
    try { response.json(await listJobs()) } catch (error) { next(error) }
  })
  app.post('/api/jobs', async (request, response, next) => {
    try {
      const problem = validateJob(request.body)
      if (problem) return response.status(400).json({ detail: problem })
      const id = randomUUID()
      const job = {
        ...request.body,
        id,
        prompt: request.body.prompt.trim(),
        duration: Number(request.body.duration),
        scene_count: request.body.scene_count ?? Math.max(2, Math.min(30, Math.ceil(Number(request.body.duration) / 5))),
        seed: Math.floor(Math.random() * 2 ** 31),
        status: 'queued',
        stage: 'Queued',
        stage_status: 'pending',
        progress: 0,
        total: 1,
        created_at: Date.now() / 1000,
        updated_at: Date.now() / 1000,
        stage_results: {},
        warning: null,
        error: null,
      }
      await persist(job)
      queue.push(job)
      runQueue()
      return response.status(202).json(job)
    } catch (error) { next(error) }
  })
  app.post('/api/jobs/:id/retry', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      if (!job) return response.status(404).json({ detail: 'Job not found.' })
      if (active.has(job.id) || job.status === 'running' || job.status === 'queued') {
        return response.status(409).json({ detail: 'Wait for this job to stop before retrying it.' })
      }
      if (!['failed', 'canceled', 'needs_review'].includes(job.status)) {
        return response.status(409).json({ detail: 'Only failed, canceled, or needs-review jobs can be retried.' })
      }
      job.status = 'queued'
      job.stage = 'Queued for retry'
      job.stage_status = 'pending'
      job.progress = 0
      job.total = 1
      job.error = null
      job.warning = null
      job.qa_report = null
      delete job.preflight_checks
      job.stage_results = {}
      job.updated_at = Date.now() / 1000
      await persist(job)
      queue.push(job)
      runQueue()
      return response.status(202).json(job)
    } catch (error) { next(error) }
  })
  app.get('/api/jobs/:id', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      return job ? response.json(job) : response.status(404).json({ detail: 'Job not found.' })
    } catch (error) { next(error) }
  })
  app.get('/api/jobs/:id/events', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      if (!job) return response.status(404).json({ detail: 'Job not found.' })
      response.set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
      response.flushHeaders()
      response.write(`event: job\ndata: ${JSON.stringify(job)}\n\n`)
      const listeners = events.get(job.id) || new Set()
      listeners.add(response)
      events.set(job.id, listeners)
      const heartbeat = setInterval(() => response.write(': heartbeat\n\n'), 15000)
      request.on('close', () => {
        clearInterval(heartbeat)
        listeners.delete(response)
      })
    } catch (error) { next(error) }
  })
  app.post('/api/jobs/:id/cancel', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      if (!job) return response.status(404).json({ detail: 'Job not found.' })
      const child = active.get(job.id)
      if (child?.pid) {
        if (process.platform === 'win32') execFile('taskkill', ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true }, (error, _stdout, stderr) => {
          if (error) void logLine(job.id, `Failed to stop worker process: ${error.message}; ${stderr || ''}`)
        })
        else {
          try { process.kill(-child.pid, 'SIGTERM') } catch (error) { if (error.code !== 'ESRCH') throw error }
        }
      }
      const index = queue.findIndex((item) => item.id === job.id)
      if (index >= 0) queue.splice(index, 1)
      job.status = 'canceled'
      job.stage = 'Canceled'
      job.stage_status = 'canceled'
      job.error = 'Canceled by user.'
      await persist(job)
      response.json(job)
    } catch (error) { next(error) }
  })
  app.post('/api/jobs/:id/scenes/:scene/retry', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      const scene = Number(request.params.scene)
      if (!job) return response.status(404).json({ detail: 'Job not found.' })
      if (job.status === 'running' || job.status === 'queued') return response.status(409).json({ detail: 'Wait for this job to finish before retrying one scene.' })
      if (!Number.isInteger(scene) || scene < 1 || scene > 100) return response.status(400).json({ detail: 'Invalid scene number.' })
      const jobFolder = path.join(jobsRoot, job.id)
      const files = [
        path.join(jobFolder, 'images', `scene-${String(scene).padStart(3, '0')}.jpg`),
        path.join(jobFolder, 'images', `scene-${String(scene).padStart(3, '0')}-final.jpg`),
        path.join(outputsRoot, job.id, 'scene-clips', `scene-${String(scene).padStart(3, '0')}.mp4`),
      ]
      await Promise.all(files.map((file) => rm(file, { force: true })))
      job.status = 'queued'
      job.stage = `Retrying scene ${scene}`
      job.stage_status = 'pending'
      job.error = null
      job.qa_report = null
      delete job.preflight_checks
      queue.push(job)
      await persist(job)
      runQueue()
      return response.status(202).json(job)
    } catch (error) { next(error) }
  })
  app.post('/api/jobs/:id/rerender', async (request, response, next) => {
    try {
      const source = await getJob(request.params.id)
      if (!source) return response.status(404).json({ detail: 'Job not found.' })
      const problem = validateJob(source)
      if (problem) return response.status(422).json({ detail: `Saved job cannot be rendered: ${problem}` })
      const id = randomUUID()
      const job = {
        ...source,
        id,
        seed: Math.floor(Math.random() * 2 ** 31),
        status: 'queued',
        stage: 'Queued',
        stage_status: 'pending',
        progress: 0,
        total: 1,
        created_at: Date.now() / 1000,
        updated_at: Date.now() / 1000,
        stage_results: {},
        error: null,
        warning: null,
        qa_report: null,
        output_file: null,
        video_url: null,
        thumbnail_url: null,
        metadata_url: null,
        assets: {},
      }
      await persist(job)
      queue.push(job)
      runQueue()
      return response.status(202).json(job)
    } catch (error) { next(error) }
  })
  app.delete('/api/jobs/:id', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      if (!job) return response.status(404).json({ detail: 'Job not found.' })
      if (active.has(job.id)) return response.status(409).json({ detail: 'Cancel the active job before deleting it.' })
      const queuedIndex = queue.findIndex((item) => item.id === job.id)
      if (queuedIndex >= 0) queue.splice(queuedIndex, 1)
      await rm(path.join(jobsRoot, job.id), { recursive: true, force: true })
      await rm(path.join(outputsRoot, job.id), { recursive: true, force: true })
      await rm(path.join(logsRoot, `job-${job.id}.log`), { force: true })
      response.status(204).end()
    } catch (error) { next(error) }
  })
  app.get('/api/jobs/:id/download', async (request, response, next) => {
    try {
      const job = await getJob(request.params.id)
      if (!job?.output_file) return response.status(404).json({ detail: 'Video is not ready.' })
      response.download(job.output_file, `prompttovideo-${job.id.slice(0, 8)}.mp4`)
    } catch (error) { next(error) }
  })
  app.use('/outputs', express.static(outputsRoot, { fallthrough: false, maxAge: '1h' }))
  app.use((error, _request, response, _next) => {
    console.error('API request failed:', error.message)
    response.status(error.status || 500).json({ detail: error.message || 'Unexpected local API error.' })
  })
  return app
}

async function start() {
  await Promise.all([mkdir(jobsRoot, { recursive: true }), mkdir(outputsRoot, { recursive: true }), mkdir(logsRoot, { recursive: true })])
  if (process.argv[2] === 'doctor') {
    const report = await doctor()
    console.log(JSON.stringify(report, null, 2))
    process.exitCode = doctorCache.status === 'ready' ? 0 : 1
    return
  }
  for (const job of await listJobs()) {
    if (job.status === 'running' || job.status === 'queued') {
      job.status = 'queued'
      job.stage = 'Recovered after restart'
      job.stage_status = 'pending'
      await persist(job)
      queue.push(job)
    }
  }
  const host = process.env.HOST || config.server.host
  const port = Number(process.env.PORT || config.server.port)
  const server = createServer(createApp())
  server.listen(port, host, () => {
    console.info(`Prompt-to-Video local API listening at http://${host}:${port}; local doctor: checking; CPUs=${os.cpus().length}`)
    void doctor().then((report) => console.info(`Local doctor finished: ${report.status}`))
    runQueue()
  })
  process.on('SIGINT', () => server.close(() => process.exit(0)))
  process.on('SIGTERM', () => server.close(() => process.exit(0)))
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await start()
}
