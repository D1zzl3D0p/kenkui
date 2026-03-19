# kenkui Windows Installer
# Run with: powershell -Command "irm https://raw.githubusercontent.com/D1zzl3D0p/kenkui/main/install.ps1 | iex"
# Or save and run: .\install.ps1

param(
    [switch]$SkipPython,
    [switch]$SkipUv,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Check-Python {
    Write-Info "Checking for Python 3.7+..."

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }

    if ($pythonCmd) {
        $version = & $pythonCmd.Source --version 2>&1
        Write-Info "Found $version"
        return $true
    }

    if ($SkipPython) {
        Write-Warn "Skipping Python check"
        return $false
    }

    Write-Warn "Python not found. Attempting to install via winget..."

    try {
        winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Info "Python installed successfully"
        return $true
    }
    catch {
        Write-Error "Failed to install Python. Please install Python 3.7+ manually from https://python.org"
        return $false
    }
}

function Check-Uv {
    Write-Info "Checking for uv..."

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        Write-Info "uv already installed"
        return $true
    }

    if ($SkipUv) {
        Write-Warn "Skipping uv check"
        return $false
    }

    Write-Warn "uv not found. Attempting to install..."

    # Try winget first
    try {
        winget install -e --id Astral.UV --silent --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            Write-Info "uv installed successfully via winget"
            return $true
        }
    }
    catch {
        Write-Warn "winget install failed, trying pip..."
    }

    # Fallback to pip
    try {
        python -m pip install uv
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            Write-Info "uv installed successfully via pip"
            return $true
        }
    }
    catch {
        Write-Error "Failed to install uv"
        return $false
    }

    return $false
}

function Install-Kenkui {
    Write-Info "Installing kenkui..."

    try {
        uv tool install kenkui
        Write-Info "kenkui installed successfully!"
        Write-Info "Run 'kenkui --help' to get started"
        return $true
    }
    catch {
        Write-Error "Failed to install kenkui: $_"
        return $false
    }
}

function Install-MultiVoice {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  Optional: Multi-Voice Support (BookNLP)" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Info "Multi-voice mode uses BookNLP to identify characters"
    Write-Info "and assign each one a different voice."
    Write-Host ""
    Write-Warn "Note: BookNLP installs ~500MB-1.5GB of NLP models."
    Write-Host ""

    $response = Read-Host "Install multi-voice support? [y/N]"
    if ($response -match '^[yY]') {
        Write-Info "Installing kenkui[multivoice]..."
        try {
            uv tool install "kenkui[multivoice]" --force
            Write-Info "Multi-voice dependencies installed."
            Write-Info "Downloading spaCy language model..."
            python -m spacy download en_core_web_sm
            Write-Info "spaCy model installed. Multi-voice support is ready!"
        }
        catch {
            Write-Warn "Multi-voice install encountered an error."
            Write-Warn "You can retry later with:"
            Write-Warn "  pip install kenkui[multivoice]"
            Write-Warn "  python -m spacy download en_core_web_sm"
        }
    }
    else {
        Write-Info "Skipping multi-voice support."
        Write-Info "To install later, run:"
        Write-Info "  pip install kenkui[multivoice]"
        Write-Info "  python -m spacy download en_core_web_sm"
    }
}

function Main {
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "  kenkui Windows Installer" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""

    # Check for winget
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wingetCmd) {
        Write-Error "winget not found. Please install App Installer from Microsoft Store or manually install Python and uv."
        Write-Host "Alternative: Install Python from https://python.org, then run: pip install uv"
        exit 1
    }

    $pythonOk = Check-Python
    $uvOk = Check-Uv

    if (-not $pythonOk -or -not $uvOk) {
        Write-Warn "Prerequisites not met. Install manually and try again."
        exit 1
    }

    $installOk = Install-Kenkui

    if ($installOk) {
        Install-MultiVoice
        Write-Host ""
        Write-Info "Installation complete!"
    }
    else {
        Write-Error "Installation failed"
        exit 1
    }
}

Main
