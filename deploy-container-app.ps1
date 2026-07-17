param(
[string]$ResourceGroup = "riskmapp-rg",
[string]$Location = "francecentral",
[string]$AppName = "riskmapp",
[string]$EnvironmentName = "riskmapp-env"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Azure CLI..."

az version | Out-Null

Write-Host "Signing in to Azure..."

az login

Write-Host "Installing or updating the Azure Container Apps extension..."

az extension add `
--name containerapp `
--upgrade `
--yes

Write-Host "Registering required Azure providers..."

az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

Write-Host "Creating resource group..."

az group create `
--name $ResourceGroup `
--location $Location `
| Out-Null

Write-Host "Building and deploying RiskMapp..."

az containerapp up `
--name $AppName `
--resource-group $ResourceGroup `
--location $Location `
--environment $EnvironmentName `
--source . `
--ingress external `
--target-port 8501

$Fqdn = az containerapp show `
--name $AppName `
--resource-group $ResourceGroup `
--query properties.configuration.ingress.fqdn `
--output tsv

Write-Host ""
Write-Host "Deployment completed."
Write-Host "Application URL: https://$Fqdn"