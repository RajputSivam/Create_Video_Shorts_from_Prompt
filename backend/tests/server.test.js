import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'

const sandbox = await mkdtemp(path.join(os.tmpdir(), 'prompttovideo-api-test-'))
process.env.JOBS_DIR = path.join(sandbox, 'jobs')
process.env.OUTPUT_DIR = path.join(sandbox, 'outputs')
const { createApp } = await import('../server.js')
const server = createApp().listen(0, '127.0.0.1')
await new Promise((resolve, reject) => {
  server.once('listening', resolve)
  server.once('error', reject)
})
const baseUrl = `http://127.0.0.1:${server.address().port}`

test('local API lists jobs and rejects invalid job inputs', async () => {
  const listing = await fetch(`${baseUrl}/api/jobs`)
  assert.equal(listing.status, 200)
  assert.deepEqual(await listing.json(), [])

  const response = await fetch(`${baseUrl}/api/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: 'shorts', prompt: 'tiny', duration: 45 }),
  })
  assert.equal(response.status, 400)
  assert.match((await response.json()).detail, /Prompt must be/)

  const invalidScenes = await fetch(`${baseUrl}/api/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'shorts', prompt: 'Coffee history with ordered beats', duration: 30,
      scene_count: 31, language: 'Hindi', voice: 'female', style: 'storybook',
    }),
  })
  assert.equal(invalidScenes.status, 400)
  assert.match((await invalidScenes.json()).detail, /Scene count/)
})

test('SSE sends the current persisted job immediately', async () => {
  const id = randomUUID()
  const folder = path.join(process.env.JOBS_DIR, id)
  await mkdir(folder, { recursive: true })
  await writeFile(path.join(folder, 'job.json'), JSON.stringify({ id, status: 'running', stage: 'images' }))

  const response = await fetch(`${baseUrl}/api/jobs/${id}/events`)
  assert.equal(response.status, 200)
  const reader = response.body.getReader()
  const first = await reader.read()
  await reader.cancel()
  const event = new TextDecoder().decode(first.value)
  assert.match(event, /event: job/)
  assert.match(event, new RegExp(id))

  const retry = await fetch(`${baseUrl}/api/jobs/${id}/retry`, { method: 'POST' })
  assert.equal(retry.status, 409)
})

after(async () => {
  await new Promise((resolve) => server.close(resolve))
  await rm(sandbox, { recursive: true, force: true })
})
