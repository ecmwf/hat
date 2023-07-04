# Pulls code from github
# (this script is not meant to be run directly but instead called by either 'hat-latest.sh' or 'hat-stable.sh'

export BRANCH_NAME=dev

# If the branch does not exist locally, create it and set the upstream
if ! git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "Branch '$BRANCH_NAME' does not exist locally, creating and setting upstream..."
  # Fetch the branch from the remote and set the upstream
  git fetch origin "$BRANCH_NAME:$BRANCH_NAME"
  git branch --set-upstream-to="origin/$BRANCH_NAME" "$BRANCH_NAME"
fi

# Checkout the specified branch quietly
git checkout "$BRANCH_NAME" --quiet

# Pull from specified branch
git pull