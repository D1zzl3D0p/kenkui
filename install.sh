#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            echo "debian"
        elif command -v dnf &> /dev/null; then
            echo "fedora"
        elif command -v pacman &> /dev/null; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "freebsd"
    else
        echo "unknown"
    fi
}

check_python() {
    log_info "Checking for Python 3.7+..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python 3.7 or later."
        return 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')
    
    if [[ "$PYTHON_MAJOR" -lt 3 ]] || ([[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 7 ]]); then
        log_error "Python 3.7+ required, found $PYTHON_VERSION"
        return 1
    fi
    
    log_info "Python $PYTHON_VERSION found"
    return 0
}

check_uv() {
    log_info "Checking for uv..."
    
    if command -v uv &> /dev/null; then
        log_info "uv already installed"
        return 0
    fi
    
    log_warn "uv not found. Installing..."
    
    case "$OS" in
        macos)
            if command -v brew &> /dev/null; then
                log_info "Installing uv via Homebrew..."
                brew install uv
            else
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                brew install uv
            fi
            ;;
        debian|linux)
            log_info "Installing uv via pip..."
            $PYTHON_CMD -m pip install uv
            ;;
        fedora)
            log_info "Installing uv via pip..."
            $PYTHON_CMD -m pip install uv
            ;;
        arch)
            log_info "Installing uv via pacman..."
            sudo pacman -S uv
            ;;
        freebsd)
            log_info "Installing uv via pkg..."
            sudo pkg install uv
            ;;
        *)
            log_info "Installing uv via curl..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            ;;
    esac
    
    if command -v uv &> /dev/null; then
        log_info "uv installed successfully"
    else
        log_error "Failed to install uv"
        return 1
    fi
    
    return 0
}

install_kenkui() {
    log_info "Installing kenkui..."
    
    if uv tool install kenkui; then
        log_info "kenkui installed successfully!"
        log_info "Run 'kenkui --help' to get started"
    else
        log_error "Failed to install kenkui"
        return 1    fi
    
    return 0
}

main() {
    log_info "Starting kenkui installer..."
    
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    check_python || exit 1
    check_uv || exit 1
    install_kenkui || exit 1
    
    log_info "Installation complete!"
}

main "$@"
