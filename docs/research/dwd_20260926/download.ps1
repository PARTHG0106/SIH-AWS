param([Parameter(Mandatory=$true)][string]$Url,
      [Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$repoRoot = 'C:\Users\Admin\Desktop\SIH-Automatic_Weather_Stations'
$docRoot = Join-Path $repoRoot 'docs\research\dwd_20260926'
$rawRoot = Join-Path $repoRoot 'data\raw\dwd_verified_research'
$target = [System.IO.Path]::GetFullPath($Destination)
if (-not ($target.StartsWith($docRoot + '\') -or $target.StartsWith($rawRoot + '\'))) {
    throw 'Research destination must stay in its designated directories.'
}
$manifestPath = Join-Path $docRoot 'retrieval_manifest.json'
if (Test-Path -LiteralPath $target) { throw 'Already archived; do not overwrite immutable research files.' }
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
$response = Invoke-WebRequest -Uri $Url -OutFile $target -PassThru -TimeoutSec 60
$entry = [ordered]@{
    url = $Url
    retrieved_at_utc = [DateTime]::UtcNow.ToString('o')
    path = $target.Substring($repoRoot.Length + 1).Replace('\','/')
    bytes = (Get-Item -LiteralPath $target).Length
    sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    status_code = [int]$response.StatusCode
    transport = 'HTTPS; Windows system certificate validation enabled via Invoke-WebRequest'
    content_type = ($response.Headers['Content-Type'] -join ',')
    last_modified_header = ($response.Headers['Last-Modified'] -join ',')
}
$manifest = @()
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = @(Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
}
$manifest += $entry
ConvertTo-Json -InputObject $manifest -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$entry | ConvertTo-Json
