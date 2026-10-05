param(
  [switch]$InstallML,
  [switch]$Models
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path '.venv\Scripts\python.exe')) {
  py -3.12 -m venv .venv
}
$python = Join-Path $root '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
& $python -m pip install -r backend\requirements.txt

if ($InstallML) {
  & $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host 'Install Ollama for Windows from https://ollama.com/download/windows, then rerun this script.' -ForegroundColor Yellow
} elseif ($Models) {
  ollama pull qwen2.5:7b
}

$directories = @('outputs', 'jobs', 'cache', 'cache\huggingface', 'cache\whisper', 'models\voices', 'assets\music', 'assets\sfx', 'assets\fonts')
foreach ($directory in $directories) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }

function Get-ModelFile([string]$url, [string]$destination) {
  if (-not (Test-Path $destination)) {
    Write-Host "Downloading $destination"
    Invoke-WebRequest -Uri $url -OutFile $destination
  }
}

Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx' 'models\voices\en_US-amy-medium.onnx'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json' 'models\voices\en_US-amy-medium.onnx.json'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx' 'models\voices\en_US-joe-medium.onnx'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json' 'models\voices\en_US-joe-medium.onnx.json'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx' 'models\voices\hi_IN-priyamvada-medium.onnx'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json' 'models\voices\hi_IN-priyamvada-medium.onnx.json'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx' 'models\voices\hi_IN-rohan-medium.onnx'
Get-ModelFile 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json' 'models\voices\hi_IN-rohan-medium.onnx.json'

if ($Models) {
  & $python -m workers.download_models
  Write-Host 'SDXL-Turbo weights are downloaded on first image-generation run and cached locally.'
}
Write-Host 'Setup dependencies and Piper voices are ready. Run: npm run doctor'
