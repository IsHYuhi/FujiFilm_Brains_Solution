echo "-----------------------------------------"
for arg in "$@"; do
    echo "$arg"
    isort $arg
    black $arg
    mypy $arg
    flake8 $arg
    echo "-----------------------------------------"
done