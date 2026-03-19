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
    Write-Info "Checking for Python 3.12+..."

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }

    if ($pythonCmd) {
        $version = & $pythonCmd.Source --version 2>&1
        $major = & $pythonCmd -c "import sys; print(sys.version_info.major)" 2>&1
        $minor = & $pythonCmd -c "import sys; print(sys.version_info.minor)" 2>&1
        Write-Info "Found Python $major.$minor ($version)"
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
            Write-Error "Python 3.12+ required, found $major.$minor"
            if (-not $SkipPython) {
                return $false
            }
        }
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
        Write-Error "Failed to install Python. Please install Python 3.12+ manually from https://python.org"
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

    $scriptRoot = $PSScriptRoot
    if (-not $scriptRoot) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    }

    $pyproject = Join-Path $scriptRoot "pyproject.toml"
    if (-not (Test-Path $pyproject)) {
        Write-Error "pyproject.toml not found. Run this script from the kenkui root directory."
        return $false
    }

    try {
        Push-Location $scriptRoot
        uv pip install -e .
        Pop-Location
        Write-Info "kenkui installed successfully!"
        Write-Info "Run 'kenkui --help' to get started"
        return $true
    }
    catch {
        Pop-Location $null
        Write-Error "Failed to install kenkui: $_"
        return $false
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
        Write-Error "winget not found. Please install App Installer from Microsoft Store."
        Write-Host "Alternative: Install Python 3.12 from https://python.org, then run:"
        Write-Host "  pip install uv"
        Write-Host "  uv pip install -e ."
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
        Write-Host ""
        Write-Info "Installation complete!"
        Write-Host ""
        Write-Info "To add a book:"
        Write-Info "  kenkui add C:\path\to\book.epub"
        Write-Host ""
        Write-Info "For multi-voice narration (BookNLP):"
        Write-Info "  BookNLP and spaCy are included by default."
        Write-Info "  The spaCy language model (en_core_web_sm) is installed automatically"
        Write-Info "  when kenkui is first used in multi-voice mode."
    }
    else {
        Write-Error "Installation failed"
        exit 1
    }
}

Main
