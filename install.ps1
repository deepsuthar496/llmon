# llmon one-click installer — Windows (PowerShell 5.1+).
# Run as the normal user (no admin needed):
#   irm https://raw.githubusercontent.com/deepsuthar496/llmon/main/install.ps1 | iex
# Or from a local checkout:
#   .\install.ps1

$ErrorActionPreference = "Stop"
$AppBin  = if ($env:APP_BIN) { $env:APP_BIN } else { "llmon" }
$Repo    = $env:LLMON_REPO     # e.g. "yourname/llmon" — enables prebuilt fast path
$Version = if ($env:LLMON_VERSION) { $env:LLMON_VERSION } else { "0.1.0" }
$InstallDir = Join-Path $env:USERPROFILE ".llmon\bin"

function Info($m) { Write-Host "[llmon] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[llmon] $m" -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$Bin = Join-Path $InstallDir "$AppBin.exe"

# 1. Prebuilt fast path.
if ($Repo) {
    $Url = "https://github.com/$Repo/releases/download/v$Version/$AppBin-$Version-x86_64-pc-windows-msvc.zip"
    Info "trying prebuilt binary: $Url"
    try {
        $zip = Join-Path $env:TEMP "llmon.zip"
        Invoke-WebRequest -Uri $Url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $env:TEMP -Force
        Copy-Item (Join-Path $env:TEMP "$AppBin.exe") $Bin -Force
        Info "installed prebuilt $Bin"
        exit 0
    } catch { Warn "no prebuilt binary — falling back to source build" }
}

# 2. Rust toolchain.
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Info "installing Rust via rustup..."
    $init = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $init
    & $init -y --profile minimal --default-toolchain stable
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# 3. Source build.
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $SrcDir "Cargo.toml"))) {
    if ($Repo) {
        Info "cloning source..."
        Remove-Item -Recurse -Force "$env:TEMP\llmon-src" -ErrorAction SilentlyContinue
        git clone --depth 1 "https://github.com/$Repo" "$env:TEMP\llmon-src"
        $SrcDir = "$env:TEMP\llmon-src"
    } else { throw "Cargo.toml not found and LLMON_REPO is unset." }
}
Info "building from source (release)..."
Push-Location $SrcDir
cargo build --release --locked
Pop-Location
Copy-Item (Join-Path $SrcDir "target\release\$AppBin.exe") $Bin -Force

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Warn "added $InstallDir to user PATH (restart terminal to take effect)"
}
Info "installed $Bin"
& $Bin bench --tokens 32
Info "done. Start with: $AppBin serve  |  $AppBin run demo 'hello'"
