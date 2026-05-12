$ErrorActionPreference='Continue'
$csv='s:\科技文献\literature_table_top7.csv'
$base='s:\科技文献\_arxiv_top7_refine'
New-Item -ItemType Directory -Force -Path $base | Out-Null
$rows=Import-Csv $csv
$datasetPatterns=@('CLINC150','BANKING77','StackOverflow','SNIPS','ATIS','MASSIVE','Jigsaw','Civil Comments','HateXplain','OLID','HASOC','COLD','Twitter','Reddit','tweet','multilingual','multimodal','research-paper stream','chronological tweet stream','chronologically-ordered tweet stream')
$metricPatterns=@('Accuracy','ACC','F1','Macro-F1','Micro-F1','Precision','Recall','AUC','AUROC','AUPRC','NMI','ARI','ECE','FPR95','Forgetting','BWT','FWT','Regret','cost','selective classification')
$out=@()
foreach($r in $rows){
  if($r.homepages -notmatch '/abs/([^/]+)$'){ continue }
  $id=$matches[1]
  $dir=Join-Path $base ($id -replace '[^a-zA-Z0-9\.-]','_')
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $archive=Join-Path $dir 'src.tar'
  $content=''
  $sourceAvailable=$true
  if(-not (Test-Path $archive)){
    try{ Invoke-WebRequest -Uri ("https://arxiv.org/e-print/{0}" -f $id) -OutFile $archive -TimeoutSec 120 | Out-Null } catch { $sourceAvailable=$false }
  }
  if(Test-Path $archive){
    try { tar -xf $archive -C $dir 2>$null } catch {}
    $texFiles=Get-ChildItem -Path $dir -Recurse -File -Include *.tex -ErrorAction SilentlyContinue
    if($texFiles.Count -gt 0){ $content=($texFiles | Sort-Object Length -Descending | ForEach-Object { Get-Content -Raw -Path $_.FullName -ErrorAction SilentlyContinue }) -join "`n" }
  }
  $abs='N/A'
  if($content -match '(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}'){
    $abs=$matches[1]
    $abs=($abs -replace '\\[a-zA-Z]+\{?',' ' -replace '[\{\}]',' ' -replace '\s+',' ').Trim()
    if($abs.Length -gt 1000){ $abs=$abs.Substring(0,1000) }
  }
  $challenge='N/A'
  if($content -match '(?is)(challenge[s]?|key challenge|main challenge|we address).*?[\.!?]'){
    $challenge=($matches[0] -replace '\\[a-zA-Z]+',' ' -replace '[\{\}]',' ' -replace '\s+',' ').Trim()
    if($challenge.Length -gt 260){ $challenge=$challenge.Substring(0,260) }
  } elseif($abs -ne 'N/A'){
    $challenge=($abs.Split('.'))[0].Trim()
  }
  $ds=@(); foreach($p in $datasetPatterns){ if($content -match [regex]::Escape($p)){ $ds+=$p } }
  $mt=@(); foreach($p in $metricPatterns){ if($content -match [regex]::Escape($p)){ $mt+=$p } }
  $out += [pscustomobject]@{
    title=$r.title
    arxiv_id=$id
    source_available=if($sourceAvailable){'yes'}else{'no'}
    challenge_excerpt=$challenge
    datasets_found=if($ds.Count){($ds|Select-Object -Unique)-join '; '}else{'N/A'}
    metrics_found=if($mt.Count){($mt|Select-Object -Unique)-join '; '}else{'N/A'}
    abstract_excerpt=$abs
  }
}
$outPath='s:\科技文献\top7_source_extract.csv'
$out | Export-Csv -Path $outPath -NoTypeInformation -Encoding UTF8
Write-Output ("done -> " + $outPath)
