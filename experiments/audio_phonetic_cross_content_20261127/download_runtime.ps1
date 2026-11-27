$ErrorActionPreference='Stop'
$dir=Join-Path (Get-Location) '.runtime/phonetic_wheels'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$packages=@{
  'transformers'='4.46.3';'tokenizers'='0.20.3';'huggingface-hub'='0.26.5';'safetensors'='0.4.5';
  'filelock'='3.16.1';'regex'='2024.11.6';'requests'='2.32.3';'tqdm'='4.67.1';'fsspec'='2024.12.0';
  'typing-extensions'='4.12.2';'urllib3'='2.2.3';'charset-normalizer'='3.4.0';'idna'='3.10';'certifi'='2024.8.30';'colorama'='0.4.6';
  'espeakng-loader'='0.2.4'
}
$manifest=@()
foreach($item in $packages.GetEnumerator()|Sort-Object Name) {
  for($attempt=0;$attempt -lt 4;$attempt++) {
    try {$meta=Invoke-RestMethod -Uri "https://pypi.org/pypi/$($item.Key)/$($item.Value)/json" -MaximumRetryCount 2 -RetryIntervalSec 2;break}
    catch {if($attempt -eq 3){throw};Start-Sleep -Seconds 1}
  }
  $choices=@($meta.urls|Where-Object {$_.filename -match '\.whl$' -and ($_.filename -match '(py3|py2.py3)-none-(any|win_amd64)\.whl$' -or $_.filename -match 'cp310-(cp310|none|abi3)-win_amd64\.whl$' -or $_.filename -match 'cp3[789]-abi3-win_amd64\.whl$')})
  if($choices.Count -eq 0) {throw "No compatible wheel for $($item.Key) $($item.Value)"}
  $wheel=$choices|Sort-Object filename|Select-Object -First 1
  $path=Join-Path $dir $wheel.filename
  $ok=$false
  if(Test-Path -LiteralPath $path) {$ok=((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant() -eq $wheel.digests.sha256)}
  if(!$ok) {
    for($attempt=0;$attempt -lt 4;$attempt++) {
      try {Invoke-WebRequest -Uri $wheel.url -OutFile $path -MaximumRetryCount 2 -RetryIntervalSec 2;break}
      catch {if($attempt -eq 3){throw};Start-Sleep -Seconds 1}
    }
  }
  $sha=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if($sha -ne $wheel.digests.sha256) {throw "Wheel hash mismatch $($wheel.filename)"}
  $manifest+=@{package=$item.Key;version=$item.Value;filename=$wheel.filename;sha256=$sha;requires_dist=$meta.info.requires_dist}
  Write-Output "WHEEL_VERIFIED $($item.Key) $($item.Value)"
}
$manifest|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $dir 'manifest.json') -Encoding utf8
Write-Output 'RUNTIME_DOWNLOAD_COMPLETE'
