import { useEffect, useState } from 'react'
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  Clapperboard,
  Clock3,
  Film,
  FilmIcon,
  History,
  LoaderCircle,
  Menu,
  Mic2,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from 'lucide-react'
import { cancelJob, createJob, deleteJob, getHealth, getJob, listJobs, rerenderJob, retryJob, retryScene } from './api.js'

const styleOptions = [
  { value: 'cinematic', label: 'Cinematic', note: 'Film-grade light & color' },
  { value: 'storybook', label: 'Storybook', note: 'Painted, full of wonder' },
  { value: 'realistic', label: 'Realistic', note: 'Natural, documentary feel' },
  { value: 'cartoon', label: 'Cartoon', note: 'Bright animated scenes' },
  { value: 'anime', label: 'Anime', note: 'Expressive cel-shaded scenes' },
]

function formatTime(seconds) {
  if (seconds < 60) return `${seconds} sec`
  return `${Math.floor(seconds / 60)} min`
}

function estimateRemaining(job) {
  if (job.status === 'queued') return 'Waiting for the next render slot'
  const elapsed = (Date.now() - Date.parse(job.stage_started_at || job.updated_at)) / 1000
  if (!Number.isFinite(elapsed) || elapsed <= 0 || job.progress <= 0 || job.total <= job.progress) return ''
  const seconds = Math.max(0, Math.round((elapsed / job.progress) * (job.total - job.progress)))
  return seconds < 60 ? `About ${seconds}s remaining` : `About ${Math.ceil(seconds / 60)} min remaining`
}

function summarizeImageFailure(failure) {
  const status = failure.match(/^([^:]+): HTTP (\d{3})/)
  if (status) {
    const reasons = {
      401: 'API key missing or invalid',
      402: 'payment or usage limit reached',
      403: 'access or quota unavailable',
      404: 'model or endpoint unavailable',
      408: 'request timed out',
      429: 'rate limit or provider quota exceeded',
    }
    return `${status[1]}: HTTP ${status[2]} (${reasons[status[2]] || 'provider temporarily unavailable'})`
  }
  return failure.replace(/https?:\/\/\S+/g, 'provider docs').slice(0, 180)
}

function App() {
  const [page, setPage] = useState('home')
  const [mode, setMode] = useState(null)
  const [jobs, setJobs] = useState([])
  const [activeJob, setActiveJob] = useState(null)
  const [loadingJobs, setLoadingJobs] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)
  const [health, setHealth] = useState(null)

  async function refreshJobs() {
    try {
      setJobs(await listJobs())
    } catch {
      setJobs([])
    } finally {
      setLoadingJobs(false)
    }
  }

  useEffect(() => {
    refreshJobs()
    const timer = window.setInterval(refreshJobs, 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let cancelled = false
    const refreshHealth = async () => {
      try {
        const result = await getHealth()
        if (!cancelled) setHealth(result)
      } catch (error) {
        if (!cancelled) {
          setHealth({
            status: 'unavailable',
            checks: { api: { status: 'failed', detail: error.message } },
          })
        }
      }
    }
    refreshHealth()
    const timer = window.setInterval(refreshHealth, 15000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [page, mode, activeJob?.id])

  useEffect(() => {
    if (!activeJob || !['queued', 'running'].includes(activeJob.status)) return undefined
    let cancelled = false
    const poll = async () => {
      try {
        const updated = await getJob(activeJob.id)
        if (!cancelled) {
          setActiveJob(updated)
          setJobs((current) => current.map((job) => job.id === updated.id ? updated : job))
        }
      } catch {
        // A transient polling failure should not interrupt an in-flight render.
      }
    }
    poll()
    const timer = window.setInterval(poll, 10000)
    const stream = new EventSource(`/api/jobs/${activeJob.id}/events`)
    stream.addEventListener('job', (event) => {
      if (cancelled) return
      const updated = JSON.parse(event.data)
      setActiveJob(updated)
      setJobs((current) => current.map((job) => job.id === updated.id ? updated : job))
    })
    stream.addEventListener('stage', (event) => {
      if (cancelled) return
      const progress = JSON.parse(event.data)
      setActiveJob((current) => {
        if (!current || current.id !== progress.job_id) return current
        const assets = { ...(current.assets || {}) }
        const stageResults = { ...(current.stage_results || {}) }
        stageResults[progress.stage] = { status: progress.status, detail: progress.detail || null }
        if (progress.scene) {
          assets.images = [...(assets.images || []).filter((item) => item.scene !== progress.scene.scene), progress.scene].sort((a, b) => a.scene - b.scene)
        }
        return { ...current, stage: progress.stage, stage_status: progress.status, progress: progress.progress, total: progress.total, stage_results: stageResults, assets }
      })
    })
    return () => {
      cancelled = true
      window.clearInterval(timer)
      stream.close()
    }
  }, [activeJob?.id, activeJob?.status])

  function openCreator(selectedMode) {
    setActiveJob(null)
    setMode(selectedMode)
    setPage('create')
  }

  function openJob(job) {
    setActiveJob(job)
    setMode(job.mode)
    setPage('create')
  }

  function navigate(nextPage) {
    setPage(nextPage)
    setMode(null)
    setMobileNav(false)
  }

  async function handleSubmit(payload) {
    const job = await createJob(payload)
    setActiveJob(job)
    refreshJobs()
  }

  async function handleRetry() {
    if (!activeJob) return
    const job = await retryJob(activeJob.id)
    setActiveJob(job)
    refreshJobs()
  }

  async function handleCancel() {
    if (!activeJob) return
    const updated = await cancelJob(activeJob.id)
    setActiveJob(updated)
    refreshJobs()
  }

  async function handleRetryScene(scene) {
    if (!activeJob) return
    const updated = await retryScene(activeJob.id, scene)
    setActiveJob(updated)
    refreshJobs()
  }

  async function handleDelete(job) {
    if (!window.confirm('Delete this video and its local files?')) return
    await deleteJob(job.id)
    setJobs((current) => current.filter((item) => item.id !== job.id))
    if (activeJob?.id === job.id) {
      setActiveJob(null)
      navigate('videos')
    }
  }

  async function handleRerender(job) {
    const rerendered = await rerenderJob(job.id)
    setActiveJob(rerendered)
    setMode(rerendered.mode)
    setPage('create')
    refreshJobs()
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <button className="brand" onClick={() => navigate('home')} aria-label="PromptToVideo home">
          <span className="brand-mark"><Clapperboard size={19} strokeWidth={2.3} /></span>
          <span>prompt<span className="brand-light">to</span>video</span>
        </button>
        <div className="workspace-switcher">
          <div className="workspace-avatar">S</div>
          <div className="workspace-text"><strong>Studio space</strong><small>Local workspace</small></div>
          <ChevronDown size={15} />
        </div>
        <span className="nav-label">WORKSPACE</span>
        <nav className="side-nav" aria-label="Main navigation">
          <button className={`nav-item ${page !== 'videos' ? 'nav-active' : ''}`} onClick={() => navigate('home')}><Sparkles size={17} /> Create</button>
          <button className={`nav-item ${page === 'videos' ? 'nav-active' : ''}`} onClick={() => navigate('videos')}><History size={17} /> My videos <span className="nav-count">{jobs.length}</span></button>
        </nav>
        <div className="sidebar-spacer" />
        <div className="usage-panel">
          <div className="usage-heading"><span>Local workspace</span><CircleHelp size={14} /></div>
          <div className="usage-track"><span /></div>
          <p>Your prompts and assets stay on this PC</p>
        </div>
        <button className="profile-row">
          <div className="profile-avatar">P</div>
          <div className="profile-name"><strong>Personal studio</strong><small>Local workspace</small></div>
          <Menu size={16} />
        </button>
      </aside>

      {mobileNav && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumbs"><span>Studio</span><span className="crumb-slash">/</span><strong>{page === 'videos' ? 'My videos' : page === 'create' ? (activeJob ? 'Your project' : 'New project') : 'Create'}</strong></div>
          <div className="topbar-right"><span className="api-indicator"><i /> Local studio</span><button className="help-button" title="Local setup and prerequisites are in README.md"><CircleHelp size={17} /></button></div>
        </header>

        {page === 'home' && <HomePage onChoose={openCreator} jobs={jobs} onOpenJob={openJob} health={health} />}
        {page === 'create' && <CreatorPage mode={mode} job={activeJob} onBack={() => navigate('home')} onSubmit={handleSubmit} onRetry={handleRetry} onCancel={handleCancel} onRetryScene={handleRetryScene} />}
        {page === 'videos' && <VideosPage jobs={jobs} loading={loadingJobs} onOpen={openJob} onCreate={() => navigate('home')} onDelete={handleDelete} onRerender={handleRerender} />}
      </main>
    </div>
  )
}

function HomePage({ onChoose, jobs, onOpenJob, health }) {
  return (
    <div className="page-content home-page">
      <div className="welcome-row">
        <div className="welcome-copy"><span className="eyebrow"><span className="eyebrow-line" /> YOUR STORY, IN MOTION</span><h1>What are we<br /><em>making today?</em></h1><p>One idea in. A whole world out.</p></div>
        <div className="edition-stamp"><span>STUDIO<br />EDITION</span><b>01</b></div>
      </div>
      <section className="mode-grid" aria-label="Choose a video format">
        <button className="mode-card long-card" onClick={() => onChoose('video')}>
          <div className="mode-photo photo-long" />
          <span className="card-index">01 / LANDSCAPE</span>
          <span className="mode-card-content"><span className="mode-icon"><Film size={18} /></span><span className="mode-title">Create Video</span><span className="mode-description">A story with room to breathe. Made for the big screen.</span><span className="mode-meta"><span><Clock3 size={14} /> 1–15 minutes</span><span>16:9 <i>·</i> 1080p</span></span></span>
          <span className="card-arrow"><ArrowUpRight size={20} /></span>
        </button>
        <button className="mode-card short-card" onClick={() => onChoose('shorts')}>
          <div className="mode-photo photo-short" />
          <span className="card-index">02 / VERTICAL</span>
          <span className="mode-card-content"><span className="mode-icon mode-icon-lime"><FilmIcon size={18} /></span><span className="mode-title">Create Shorts</span><span className="mode-description">Small screen. Full feeling. Built to stop the scroll.</span><span className="mode-meta"><span><Clock3 size={14} /> 15–60 seconds</span><span>9:16 <i>·</i> 1080p</span></span></span>
          <span className="card-arrow"><ArrowUpRight size={20} /></span>
        </button>
      </section>
      <section className="recent-section">
        <div className="section-heading"><div><span className="eyebrow">THE EDIT BAY</span><h2>Recently made</h2></div><button className="text-link" onClick={() => onOpenJob && jobs[0] && onOpenJob(jobs[0])}>View latest <ArrowUpRight size={15} /></button></div>
        {jobs.length > 0 ? <div className="recent-list">{jobs.slice(0, 3).map((job) => <JobRow key={job.id} job={job} onClick={() => onOpenJob(job)} />)}</div> : <div className="empty-recent"><span className="empty-icon"><WandSparkles size={19} /></span><div><strong>Nothing in the edit bay yet</strong><p>Your finished pieces and works in progress will land here.</p></div><span className="empty-underline" /></div>}
      </section>
      <HealthChecklist health={health} />
      <div className="home-footnote"><span>LOCAL-FIRST · NO PAID API REQUIRED</span><span>YOUR IDEAS STAY ON THIS PC</span></div>
    </div>
  )
}

function HealthChecklist({ health }) {
  const items = [
    ['gpu', 'CUDA GPU'],
    ['ollama', 'Ollama local LLM'],
    ['sdxl', 'Local SDXL'],
    ['piper', 'Piper local TTS'],
    ['whisper', 'Whisper local captions'],
    ['ffmpeg', 'FFmpeg'],
    ['ffprobe', 'ffprobe'],
    ['voices', 'Offline voice models'],
    ['fonts', 'English / Hindi fonts'],
  ]
  return <section className="health-checklist" aria-label="Startup service checks">
    <div className="section-heading"><div><span className="eyebrow">SYSTEM CHECK</span><h2>Ready to create</h2></div><span className={`health-summary health-${health?.status || 'checking'}`}>{health?.status === 'ready' ? 'ALL CHECKS PASSED' : health?.status === 'checking' ? 'CHECKING SERVICES' : 'ACTION NEEDED'}</span></div>
    <ul>{items.map(([key, label]) => {
      const check = health?.checks?.[key]
      const status = check?.status || 'checking'
      const isPassed = status === 'passed' || status === 'ready'
      return <li key={key} className={`health-item health-item-${status}`}>
        <span className="health-icon">{isPassed ? <Check size={13} /> : status === 'checking' ? <LoaderCircle size={13} className="spin" /> : <X size={13} />}</span>
        <span className="health-label">{label}</span>
        <span className="health-detail">{check?.detail || 'Waiting for startup self-test.'}</span>
      </li>
    })}</ul>
  </section>
}

function CreatorPage({ mode, job, onBack, onSubmit, onRetry, onCancel, onRetryScene }) {
  const [prompt, setPrompt] = useState(job?.prompt || '')
  const [duration, setDuration] = useState(job?.duration || (mode === 'shorts' ? 45 : 120))
  const [sceneCount, setSceneCount] = useState(job?.scene_count || (mode === 'shorts' ? 6 : 10))
  const [language, setLanguage] = useState(job?.language || 'English')
  const [style, setStyle] = useState(job?.style || 'cinematic')
  const [voice, setVoice] = useState(job?.voice || 'female')
  const [musicMood, setMusicMood] = useState(job?.music_mood || 'auto')
  const [effectsLevel, setEffectsLevel] = useState(job?.effects_level || 'cinematic')
  const [quality, setQuality] = useState(job?.quality || 'balanced')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const isShort = mode === 'shorts'
  const min = isShort ? 15 : 60
  const max = isShort ? 60 : 900
  async function submit(event) {
    event.preventDefault()
    if (prompt.trim().length < 8) return
    setSubmitting(true)
    setError('')
    try {
      await onSubmit({ mode, prompt: prompt.trim(), duration, scene_count: sceneCount, language, style, voice, music_mood: musicMood, effects_level: effectsLevel, quality })
    } catch (problem) {
      setError(problem.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-content creator-page">
      <button className="back-link" onClick={onBack}><ArrowLeft size={16} /> Back to studio</button>
      <div className="creator-heading"><span className="eyebrow"><span className="eyebrow-line" /> {isShort ? 'VERTICAL FILM · 9:16' : 'LONG-FORM FILM · 16:9'}</span><h1>{job ? 'Your film is taking shape.' : <>Let’s make a<br /><em>{isShort ? 'little big thing.' : 'good story.'}</em></>}</h1><p>{job ? 'We’ll keep your progress and assets safe as the render comes together.' : 'Give us the spark. We’ll take it from there.'}</p></div>
      <div className="creator-layout">
        <form className="creation-form" onSubmit={submit}>
          <div className="form-block prompt-block"><div className="field-heading"><label htmlFor="story-prompt">The idea</label><span>{prompt.length}/3000</span></div><textarea id="story-prompt" value={prompt} maxLength={3000} onChange={(event) => setPrompt(event.target.value)} placeholder="A young astronomer discovers a signal hidden in the oldest light from space…" disabled={!!job} required minLength={8} /><div className="prompt-foot"><Sparkles size={14} /><span>Specific details make for better scenes.</span></div></div>
          <div className="form-block quality-block"><div className="field-heading"><label>Render quality</label><span>Local SDXL</span></div><div className="segmented-control">{['fast', 'balanced', 'high'].map((option) => <button type="button" key={option} className={quality === option ? 'segment-selected' : ''} onClick={() => setQuality(option)} disabled={!!job}>{option}</button>)}</div><p className="quality-note">All generation stays on this PC. Model downloads happen once.</p></div>
          <div className="form-block duration-block"><div className="field-heading"><label htmlFor="duration">Length</label><strong>{formatTime(duration)}</strong></div><input id="duration" className="range-input" type="range" min={min} max={max} step={isShort ? 1 : 60} value={duration} onChange={(event) => setDuration(Number(event.target.value))} disabled={!!job} /><div className="range-labels"><span>{isShort ? '15 sec' : '1 min'}</span><span>{isShort ? '60 sec' : '15 min'}</span></div></div>
          <div className="form-block duration-block"><div className="field-heading"><label htmlFor="scene-count">Scenes</label><strong>{sceneCount} scenes</strong></div><input id="scene-count" className="range-input" type="range" min="2" max="30" step="1" value={sceneCount} onChange={(event) => setSceneCount(Number(event.target.value))} disabled={!!job} /><div className="range-labels"><span>2 scenes</span><span>30 scenes</span></div></div>
          <div className="form-grid">
            <div className="form-block"><label className="standalone-label">Language</label><div className="segmented-control">{['English', 'Hindi'].map((option) => <button key={option} type="button" className={language === option ? 'segment-selected' : ''} onClick={() => setLanguage(option)} disabled={!!job}>{option}</button>)}</div></div>
            <div className="form-block"><label className="standalone-label">Narrator</label><div className="segmented-control">{['female', 'male'].map((option) => <button key={option} type="button" className={voice === option ? 'segment-selected' : ''} onClick={() => setVoice(option)} disabled={!!job}>{option === 'female' ? 'Female' : 'Male'}</button>)}</div></div>
          </div>
          <div className="form-block style-block"><div className="field-heading"><label htmlFor="visual-style">Visual style</label><span>Pick a look</span></div><div className="style-options" id="visual-style">{styleOptions.map((option, index) => <button className={`style-option style-swatch-${index} ${style === option.value ? 'style-selected' : ''}`} type="button" key={option.value} onClick={() => setStyle(option.value)} disabled={!!job}><span className="style-swatch" /><span className="style-copy"><strong>{option.label}</strong><small>{option.note}</small></span>{style === option.value && <Check size={16} className="style-check" />}</button>)}</div></div>
          {!job && <div className="form-grid local-options">
            <div className="form-block"><label className="standalone-label">Music mood</label><div className="segmented-control">{['auto', 'calm', 'epic', 'happy', 'sad', 'mystery'].map((option) => <button key={option} type="button" className={musicMood === option ? 'segment-selected' : ''} onClick={() => setMusicMood(option)}>{option}</button>)}</div></div>
            <div className="form-block"><label className="standalone-label">Effects level</label><div className="segmented-control">{['minimal', 'cinematic', 'dynamic'].map((option) => <button key={option} type="button" className={effectsLevel === option ? 'segment-selected' : ''} onClick={() => setEffectsLevel(option)}>{option}</button>)}</div></div>
          </div>}
          {!job && <div className="submit-row"><button className="primary-button" type="submit" disabled={submitting || prompt.trim().length < 8}>{submitting ? <><LoaderCircle className="spin" size={17} /> Starting your film</> : <><WandSparkles size={17} /> Generate {isShort ? 'short' : 'video'} <ArrowUpRight size={16} /></>}</button><span>Generated locally · no provider credits</span></div>}
          {error && <div className="form-error" role="alert"><X size={16} /> {error}</div>}
        </form>
        <aside className="preview-column">
          {job ? <JobStatus job={job} onRetry={onRetry} onCancel={onCancel} onRetryScene={onRetryScene} /> : <>
            <div className={`preview-art ${isShort ? 'preview-vertical' : ''}`}><div className="preview-photo" /><span className="preview-tag">YOUR CANVAS</span><div className="preview-art-bottom"><span className="preview-play"><span /></span><span>{isShort ? '9:16' : '16:9'} <i>·</i> {formatTime(duration)}</span></div></div>
            <div className="spec-row"><span>OUTPUT</span><strong>{isShort ? '1080 × 1920' : '1920 × 1080'} <i>MP4</i></strong></div>
            <div className="spec-row"><span>INCLUDED</span><strong>Scenes, voice, captions</strong></div>
            <div className="provider-note"><span className="note-star">✳</span><p>Local models create narration, scenes and captions on this PC.</p></div>
          </>}
        </aside>
      </div>
    </div>
  )
}

function JobStatus({ job, onRetry, onCancel, onRetryScene }) {
  const progress = job.total > 0 ? Math.min(100, Math.round((job.progress / job.total) * 100)) : 4
  const running = ['queued', 'running'].includes(job.status)
  const completed = job.status === 'completed'
  const remaining = estimateRemaining(job)
  const stages = ['preflight', 'script', 'voice', 'images', 'upscale', 'captions', 'music', 'sfx', 'effects', 'render']
  const placeholderScenes = (job.assets?.images || []).filter((image) => image.placeholder)
  const placeholderFailureGroups = placeholderScenes.reduce((groups, image) => {
    const summary = summarizeImageFailure(image.error || 'Local SDXL is not available')
    if (!groups.has(summary)) groups.set(summary, [])
    groups.get(summary).push(image.scene)
    return groups
  }, new Map())
  const qaChecks = Object.values(job.qa_report?.checks || {})
  const qaReason = qaChecks.find((item) => item.status === 'FAIL' || item.status === 'WARN')?.reason

  return <div className="job-status-panel">
    {job.qa_report && <div className={`quality-badge quality-${job.qa_report.status}`} role="status"><strong>Quality: {job.qa_report.status === 'pass' ? 'Pass' : 'Needs review'}</strong>{qaReason && <p>{qaReason}</p>}</div>}
    {completed ? <>
      <div className="status-heading"><span className="status-check"><Check size={17} /></span><div><span className="eyebrow">READY FOR ITS PREMIERE</span><h2>Your video is ready.</h2></div></div>
      <div className={`video-frame ${job.mode === 'shorts' ? 'video-frame-vertical' : ''}`}><video controls playsInline src={job.video_url} poster={job.thumbnail_url || undefined} /><span className="video-dimensions">{job.mode === 'shorts' ? '9:16' : '16:9'}</span></div>
      <a className="primary-button download-button" href={`/api/jobs/${job.id}/download`}><ArrowDownToLine size={17} /> Download MP4</a>
      <div className="spec-row"><span>FILE FORMAT</span><strong>MP4 · H.264 + AAC</strong></div>
      {job.assets?.script && <div className="video-metadata"><h3>{job.assets.script.title}</h3><p>{job.assets.script.description}</p><div>{(job.assets.script.tags || []).map((tag) => <span key={tag}>{tag}</span>)}</div></div>}
      {job.warning && <div className="job-warning" role="status"><strong>Some scenes use gradient fallbacks</strong><p>{job.warning}</p></div>}
    </> : <>
      <div className="status-heading"><span className={`status-loader ${['failed', 'canceled', 'needs_review'].includes(job.status) ? 'status-failed' : ''}`}>{['failed', 'canceled', 'needs_review'].includes(job.status) ? <X size={16} /> : <LoaderCircle size={17} className="spin" />}</span><div><span className="eyebrow">{job.status === 'failed' ? 'NEEDS A SECOND TAKE' : job.status === 'needs_review' ? 'QUALITY REVIEW REQUIRED' : job.status === 'canceled' ? 'CANCELED' : job.status === 'queued' ? 'IN THE QUEUE' : 'IN PRODUCTION'}</span><h2>{job.status === 'failed' ? 'The render paused.' : job.status === 'needs_review' ? 'This video is not marked ready.' : job.status === 'canceled' ? 'Render canceled.' : job.stage}</h2></div></div>
      {['failed', 'canceled', 'needs_review'].includes(job.status) ? <div className="failure-box"><p>{job.error || qaReason || 'Something interrupted this render.'}</p>{job.video_url && <div className={`video-frame ${job.mode === 'shorts' ? 'video-frame-vertical' : ''}`}><video controls playsInline src={job.video_url} /></div>}{['failed', 'needs_review'].includes(job.status) && <button className="primary-button" onClick={onRetry}><RotateCcw size={16} /> Retry with these settings</button>}</div> : <>
        <div className="progress-card"><div className="progress-label"><span>{job.stage || 'Getting everything ready'}</span><strong>{job.total > 0 ? `${job.progress}/${job.total}` : '…'}</strong></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><div className="progress-caption"><span>{remaining || 'Progress saves as your film grows'}</span><span>{job.total > 0 ? `${progress}%` : ''}</span></div></div>
        <div className="pipeline-list">{stages.map((stage, index) => {
          const result = job.stage_results?.[stage]
          const current = job.stage === stage
          const label = stage === 'sfx' ? 'Sound effects' : stage === 'images' ? 'Images' : stage.charAt(0).toUpperCase() + stage.slice(1)
          return <div className={`pipeline-step ${current ? 'pipeline-current' : ''}`} key={stage}><span className="pipeline-dot">{result?.status === 'completed' ? <Check size={13} /> : index + 1}</span><span>{label}</span><span className="pipeline-state">{result?.status === 'completed' ? 'DONE' : result?.status === 'degraded' ? 'WARNING' : current ? 'NOW' : ''}</span></div>
        })}</div>
        {running && <button className="secondary-button cancel-button" onClick={onCancel}><X size={15} /> Cancel render</button>}
        <p className="wait-note">Long videos can take a while. This tab can stay open while we work.</p>
      </>}
    </>}
    {job.assets?.images?.length > 0 && <div className="scene-preview-grid">{job.assets.images.map((image) => <figure key={image.scene}><img src={image.preview_url} alt={`Generated scene ${image.scene}`} /><figcaption>Scene {image.scene}{image.placeholder ? ' · gradient fallback' : ''}</figcaption></figure>)}</div>}
    {job.status !== 'completed' && job.warning && <div className="job-warning" role="status"><strong>Image generation warning</strong><p>{job.warning}</p></div>}
    {placeholderScenes.length > 0 && <div className="image-notice" role="status"><strong>{placeholderScenes.length} scene{placeholderScenes.length === 1 ? '' : 's'} using gradient fallback{placeholderScenes.length === 1 ? '' : 's'}</strong><p>No image-prompt text is drawn on fallback visuals.</p><details><summary>Provider details</summary><ul>{Array.from(placeholderFailureGroups, ([failure, scenes]) => <li key={failure}>{failure} · scenes {scenes.join(', ')}</li>)}</ul></details></div>}
    {placeholderScenes.length > 0 && completed && <div className="scene-retries"><strong>Retry a fallback scene</strong>{placeholderScenes.map((image) => <button key={image.scene} onClick={() => onRetryScene(image.scene)}><RotateCcw size={13} /> Scene {image.scene}</button>)}</div>}
  </div>
}

function VideosPage({ jobs, loading, onOpen, onCreate, onDelete, onRerender }) {
  const [filter, setFilter] = useState('all')
  const filtered = filter === 'all' ? jobs : jobs.filter((job) => job.mode === filter)
  return <div className="page-content videos-page"><div className="videos-heading"><div><span className="eyebrow"><span className="eyebrow-line" /> YOUR EDIT BAY</span><h1>My <em>videos.</em></h1><p>Every idea you’ve brought to life, all in one place.</p></div><button className="primary-button" onClick={onCreate}><Plus size={17} /> New project</button></div>
    <div className="video-toolbar"><div className="filter-tabs"><button className={filter === 'all' ? 'filter-active' : ''} onClick={() => setFilter('all')}>All <span>{jobs.length}</span></button><button className={filter === 'video' ? 'filter-active' : ''} onClick={() => setFilter('video')}>Video</button><button className={filter === 'shorts' ? 'filter-active' : ''} onClick={() => setFilter('shorts')}>Shorts</button></div><span className="sort-note"><History size={14} /> Newest first</span></div>
    <div className="video-library">{loading ? <div className="library-empty"><LoaderCircle className="spin" size={21} /> Loading your edit bay</div> : filtered.length ? filtered.map((job) => <div className="library-row" key={job.id}><JobRow job={job} onClick={() => onOpen(job)} expanded /><div className="library-actions">{job.status === 'completed' && <button className="icon-button" title="Re-render with these settings" aria-label={`Re-render ${job.prompt}`} onClick={() => onRerender(job)}><RotateCcw size={15} /></button>}<button className="icon-button" title="Delete video" aria-label={`Delete ${job.prompt}`} onClick={() => onDelete(job)}><Trash2 size={15} /></button></div></div>) : <div className="library-empty"><span className="empty-icon"><Film size={19} /></span><strong>No videos here yet</strong><p>Start with a prompt and your first film will show up here.</p><button className="text-link" onClick={onCreate}>Create your first video <ArrowUpRight size={15} /></button></div>}</div>
    <div className="library-footnote">Your videos and cached assets are stored locally in this project.</div>
  </div>
}

function JobRow({ job, onClick, expanded = false }) {
  const done = job.status === 'completed'
  const failed = ['failed', 'needs_review'].includes(job.status)
  const quality = job.qa_report?.status
  const qualityReason = Object.values(job.qa_report?.checks || {}).find((item) => item.status === 'FAIL' || item.status === 'WARN')?.reason
  return <button className={`job-row ${expanded ? 'job-row-expanded' : ''}`} onClick={onClick}>
    <span className={`job-thumb ${job.mode === 'shorts' ? 'job-thumb-short' : ''} ${done ? 'job-thumb-done' : ''}`}><span className="thumb-image" style={job.thumbnail_url ? { backgroundImage: `url("${job.thumbnail_url}")` } : undefined} /><span className="thumb-play">{done ? <Check size={14} /> : <Film size={14} />}</span></span>
    <span className="job-main"><strong>{job.prompt}</strong><span className="job-meta">{job.mode === 'shorts' ? 'Shorts' : 'Video'} <i>·</i> {formatTime(job.duration)} <i>·</i> {new Date(job.created_at).toLocaleDateString()}</span>{quality && <span className={`quality-inline ${quality === 'pass' ? 'quality-inline-pass' : 'quality-inline-review'}`}>Quality: {quality === 'pass' ? 'Pass' : 'Needs review'}</span>}{expanded && qualityReason && <span className="quality-reason">{qualityReason}</span>}</span>
    <span className={`job-status-pill ${done ? 'pill-done' : failed ? 'pill-failed' : 'pill-progress'}`}><i />{done ? 'Ready' : job.status === 'needs_review' ? 'Review' : failed ? 'Failed' : job.status === 'queued' ? 'Queued' : 'In progress'}</span>
    {expanded && <span className="job-open"><ArrowUpRight size={17} /></span>}
  </button>
}

export default App