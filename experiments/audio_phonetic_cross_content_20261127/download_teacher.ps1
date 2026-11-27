$ErrorActionPreference='Stop'
$revision='ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
$model=Join-Path (Get-Location) '.runtime/models/wav2vec2-lv-60-espeak-cv-ft'
New-Item -ItemType Directory -Force -Path $model | Out-Null
foreach($name in @('config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','special_tokens_map.json')) {
  $dest=Join-Path $model $name
  $ok=$false
  if(Test-Path -LiteralPath $dest) {try {$null=Get-Content -LiteralPath $dest -Raw|ConvertFrom-Json -AsHashtable;$ok=$true}catch{}}
  if($ok) {continue}
  $url="https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/$revision/$name"
  for($attempt=0;$attempt -lt 4;$attempt++) {
    try {Invoke-WebRequest -Uri $url -OutFile $dest -MaximumRetryCount 2 -RetryIntervalSec 2;$null=Get-Content -LiteralPath $dest -Raw|ConvertFrom-Json -AsHashtable;break}
    catch {if($attempt -eq 3){throw};Start-Sleep -Seconds 1}
  }
  Write-Output "TEACHER_CONFIG $name"
}
$weight=Join-Path $model 'pytorch_model.bin'
$size=1263535127
if(!(Test-Path -LiteralPath $weight) -or (Get-Item -LiteralPath $weight).Length -ne $size) {
  $url="https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/$revision/pytorch_model.bin?download=true"
  & curl.exe --location --fail --retry 4 --retry-delay 2 --connect-timeout 30 --continue-at - --output $weight $url
  if($LASTEXITCODE -ne 0) {throw 'Teacher download failed; partial file retained for resume'}
}
$sha=(Get-FileHash -Algorithm SHA256 -LiteralPath $weight).Hash.ToLowerInvariant()
if($sha -ne '3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e') {throw 'Official teacher SHA256 mismatch'}
@{repository='facebook/wav2vec2-lv-60-espeak-cv-ft';revision=$revision;weight_sha256=$sha;weight_bytes=$size}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $model 'revision_verified.json') -Encoding utf8
Write-Output 'TEACHER_DOWNLOAD_VERIFIED'
