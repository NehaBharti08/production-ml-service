<#
.SYNOPSIS
    Windows shim for the Makefile. Mirrors its targets one for one.

.DESCRIPTION
    `make` is not installed by default on Windows, and the Makefile is the
    canonical entrypoint for CI and for anyone cloning on Linux/macOS. Rather
    than require a GNU make install just to run the project locally, this shim
    maps the same target names onto the same commands.

    If the two ever drift apart, the Makefile wins — it is what CI executes.

.EXAMPLE
    .\make.ps1 setup
    .\make.ps1 check
    .\make.ps1 serve
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$compose     = @('docker', 'compose')
$composeFull = @('docker', 'compose', '-f', 'docker-compose.yml', '-f', 'docker-compose.monitoring.yml')

function Invoke-Step {
    param([string[]]$Command)
    Write-Host ">> $($Command -join ' ')" -ForegroundColor DarkGray
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Failed (exit $LASTEXITCODE): $($Command -join ' ')"
    }
}

function Invoke-Pytest {
    param([string]$Path, [string]$Marker)
    $env:MLSERVICE_ENV = 'test'
    try   { Invoke-Step @('uv', 'run', 'pytest', $Path, '-m', $Marker) }
    finally { Remove-Item Env:\MLSERVICE_ENV -ErrorAction SilentlyContinue }
}

switch ($Target) {

    'help' {
        Write-Host "`nTargets (mirrors the Makefile):`n" -ForegroundColor Cyan
        @(
            @('setup',            'Create venv, install deps, install git hooks'),
            @('doctor',           'Check this environment can run the service'),
            @('config',           'Print fully resolved configuration'),
            @('lint',             'Lint and format-check (no changes written)'),
            @('format',           'Auto-fix lint and formatting'),
            @('typecheck',        'Static type check'),
            @('hooks',            'Run every pre-commit hook over the whole repo'),
            @('test',             'Unit + contract + behavior'),
            @('test-unit',        'Unit tests'),
            @('test-contract',    'API schema stability'),
            @('test-behavior',    'Model invariance and directional expectations'),
            @('test-integration', 'Integration tests (needs a running container)'),
            @('test-all',         'Every suite, with coverage'),
            @('check',            'What CI runs on every PR'),
            @('serve',            'Core stack: api + mlflow + postgres'),
            @('full',             'Core stack + prometheus + grafana'),
            @('down',             'Stop everything, keep volumes'),
            @('clean',            'Stop everything and DELETE volumes'),
            @('logs',             'Tail API logs'),
            @('data',             'Phase 1 - download and verify the dataset'),
            @('audit',            'Phase 1 - reproduce the data audit'),
            @('train',            'Phase 2 - train candidates, register champion'),
            @('loadtest',         'Phase 4 - k6 load test'),
            @('drift',            'Phase 6 - induce drift and show detection'),
            @('k8s-demo',         'Phase 7 - kind cluster, canary, rollback')
        ) | ForEach-Object {
            Write-Host ("  {0,-20} {1}" -f $_[0], $_[1])
        }
        Write-Host ''
    }

    'setup' {
        Invoke-Step @('uv', 'sync', '--all-groups')
        Invoke-Step @('uv', 'run', 'pre-commit', 'install', '--install-hooks')
        Invoke-Step @('uv', 'run', 'pre-commit', 'install', '--hook-type', 'commit-msg')
        Invoke-Step @('uv', 'run', 'mlservice', 'doctor')
    }

    'doctor' { Invoke-Step @('uv', 'run', 'mlservice', 'doctor') }
    'config' { Invoke-Step @('uv', 'run', 'mlservice', 'config', '--thresholds') }

    'lint' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', 'src', 'tests')
        Invoke-Step @('uv', 'run', 'ruff', 'format', '--check', 'src', 'tests')
    }

    'format' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', '--fix', 'src', 'tests')
        Invoke-Step @('uv', 'run', 'ruff', 'format', 'src', 'tests')
    }

    'typecheck' { Invoke-Step @('uv', 'run', 'mypy', 'src') }
    'hooks'     { Invoke-Step @('uv', 'run', 'pre-commit', 'run', '--all-files') }

    'test' {
        Invoke-Pytest 'tests/unit'     'unit'
        Invoke-Pytest 'tests/contract' 'contract'
        Invoke-Pytest 'tests/behavior' 'behavior'
    }

    'test-unit'        { Invoke-Pytest 'tests/unit'        'unit' }
    'test-contract'    { Invoke-Pytest 'tests/contract'    'contract' }
    'test-behavior'    { Invoke-Pytest 'tests/behavior'    'behavior' }
    'test-integration' { Invoke-Pytest 'tests/integration' 'integration' }

    'test-all' {
        $env:MLSERVICE_ENV = 'test'
        try {
            Invoke-Step @('uv', 'run', 'pytest', '--cov', '--cov-report=term-missing', '--cov-report=xml')
        } finally { Remove-Item Env:\MLSERVICE_ENV -ErrorAction SilentlyContinue }
    }

    'check' {
        & $PSCommandPath 'lint'
        & $PSCommandPath 'typecheck'
        & $PSCommandPath 'test'
    }

    'serve' {
        Invoke-Step ($compose + @('up', '-d', '--build'))
        Write-Host 'api      http://localhost:8000/docs'
        Write-Host 'mlflow   http://localhost:5000'
    }

    'full' {
        Invoke-Step ($composeFull + @('up', '-d', '--build'))
        Write-Host 'api        http://localhost:8000/docs'
        Write-Host 'mlflow     http://localhost:5000'
        Write-Host 'prometheus http://localhost:9090'
        Write-Host 'grafana    http://localhost:3000'
    }

    'down'  { Invoke-Step ($composeFull + @('down')) }
    'clean' { Invoke-Step ($composeFull + @('down', '-v')) }
    'logs'  { Invoke-Step ($compose + @('logs', '-f', 'api')) }

    'data'     { Invoke-Step @('uv', 'run', 'mlservice', 'data', 'download') }
    'audit'    { Invoke-Step @('uv', 'run', 'mlservice', 'data', 'audit') }
    'train'    { Invoke-Step @('uv', 'run', 'mlservice', 'train', 'run') }
    'loadtest' { Invoke-Step @('k6', 'run', 'loadtest/k6/ramp.js') }
    'drift'    { Invoke-Step @('uv', 'run', 'mlservice', 'monitor', 'replay', '--induce-drift') }

    'k8s-demo' {
        # Compose and kind together exceed what a 16 GB laptop can serve.
        Invoke-Step ($composeFull + @('down'))
        Invoke-Step @('bash', 'scripts/k8s_rollback_demo.sh')
    }

    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        & $PSCommandPath 'help'
        exit 1
    }
}
