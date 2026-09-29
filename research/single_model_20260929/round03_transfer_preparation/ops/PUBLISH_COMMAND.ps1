param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/master/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$taskOps = 'C:/Users/master/Documents/ChatGPT/MTO/research_state/single_model_20260929/ops'
$env:GIT_SSH_COMMAND = 'C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o HostName=ssh.github.com -o HostKeyAlias=github.com -p 443 -o "ProxyCommand=C:/Users/master/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe C:/Users/master/Documents/ChatGPT/MTO/research_state/single_model_20260929/ops/ssh_proxy.py %h %p"'
$taskArgs = @('--archive', $Archive)
if ($Publish) { $taskArgs += '--publish' }
& $taskPython ($taskOps + '/publish_records.py') @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Publication command failed; inspect and preserve the exact local commit before retry.' }
