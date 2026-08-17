#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-research-assistant}"
IMAGE="${IMAGE:-ghcr.io/abdelrhmanebied/research-assistant:latest}"
ENV_FILE="${ENV_FILE:-.env}"
SECRET_SRC="k8s/secret.env"
SECRET_NAME="assistant-secrets"

cd "$(dirname "$0")/.."

if [[ ! -f "$ENV_FILE" ]]; then
    echo "error: $ENV_FILE not found — copy .env.example to .env and fill in your API keys" >&2
    exit 1
fi

# Build the Secret from .env (gitignored), so keys never touch git or the repo.
grep -E '^(GEMINI|OPENAI|ANTHROPIC|TAVILY)_API_KEY=' "$ENV_FILE" > "$SECRET_SRC"
trap 'rm -f "$SECRET_SRC"' EXIT

kubectl create secret generic "$SECRET_NAME" \
    --namespace "$NAMESPACE" \
    --from-env-file="$SECRET_SRC" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

kubectl set image "deployment/research-assistant" app="$IMAGE" --namespace "$NAMESPACE"

kubectl rollout status "deployment/research-assistant" --namespace "$NAMESPACE" --timeout=300s

echo "Deployed $IMAGE to $NAMESPACE"