async function request(path, options) {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return body
}

export const listJobs = () => request('/api/jobs')
export const getHealth = () => request('/api/health')
export const getJob = (id) => request(`/api/jobs/${id}`)
export const createJob = (payload) => request('/api/jobs', { method: 'POST', body: JSON.stringify(payload) })
export const retryJob = (id) => request(`/api/jobs/${id}/retry`, { method: 'POST' })
export const cancelJob = (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' })
export const retryScene = (id, scene) => request(`/api/jobs/${id}/scenes/${scene}/retry`, { method: 'POST' })
export const rerenderJob = (id) => request(`/api/jobs/${id}/rerender`, { method: 'POST' })
export const deleteJob = (id) => request(`/api/jobs/${id}`, { method: 'DELETE' })
