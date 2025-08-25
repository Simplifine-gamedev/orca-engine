#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Get GCP Project ID from gcloud config
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP project ID not found. Please run 'gcloud config set project <YOUR_PROJECT_ID>'"
    exit 1
fi

echo "🚀 Fetching secrets for project: $PROJECT_ID"

# Path for the new .env file, assuming the script is run from the `backend` directory or project root.
# It will place the .env file in the `backend` directory.
ENV_FILE_PATH="backend/.env"

# List all secrets in the project
SECRET_NAMES=$(gcloud secrets list --project="$PROJECT_ID" --format="value(name)")

if [ -z "$SECRET_NAMES" ]; then
    echo "🤔 No secrets found for project $PROJECT_ID."
    exit 0
fi

# Create or clear the .env file
> "$ENV_FILE_PATH"

echo "📝 Creating .env file at $ENV_FILE_PATH"

# Loop through each secret name, fetch its value, and append to the .env file
for SECRET_NAME in $SECRET_NAMES; do
    echo "   -> Fetching ${SECRET_NAME}..."
    # Access the latest version of the secret
    SECRET_VALUE=$(gcloud secrets versions access latest --project="$PROJECT_ID" --secret="$SECRET_NAME")
    
    # Write KEY=VALUE to the .env file. No quotes.
    echo "${SECRET_NAME}=${SECRET_VALUE}" >> "$ENV_FILE_PATH"
done

echo "✅ Successfully created .env file with $(echo "$SECRET_NAMES" | wc -w | xargs) secrets."
echo "Location: $ENV_FILE_PATH"
