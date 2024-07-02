set -e
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

cargo run --bin dev-tasks-runner -- $@