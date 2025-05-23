SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
model="google/siglip2-base-patch16-512"

python $SCRIPT_DIR/train.py --vision_model_name $model