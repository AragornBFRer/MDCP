from pathlib import Path
from typing import Union

RANDOM_SEED = 3
# 1,2,3,... for new version test (major updates)
# 8 for standard evaluations (for publication)
# 19 for quick dirty trials

TRUE_TRAIN_RATIO = 0.375
TRUE_CAL_RATIO = 0.125
TRUE_TEST_RATIO = 0.5

MIMIC_CALIBRATION_RATIO = 0.5
MIMIC_TEST_RATIO = 0.5

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_OUT_BASE = Path("..") / "eval_out"


PathLike = Union[Path, str]


def _to_path(path: PathLike) -> Path:
	return path if isinstance(path, Path) else Path(path)


def resolve_project_path(path: PathLike) -> Path:
	path_obj = _to_path(path)
	if path_obj.is_absolute():
		return path_obj
	return (BASE_DIR / path_obj).resolve()


def ensure_project_dir(path: PathLike) -> Path:
	target = resolve_project_path(path)
	target.mkdir(parents=True, exist_ok=True)
	return target


def prefer_relative_path(path: PathLike) -> Path:
	path_obj = resolve_project_path(path)
	try:
		relative = path_obj.relative_to(PROJECT_ROOT)
		return Path("..") / relative
	except ValueError:
		return path_obj


def eval_out_relative(*parts: str) -> Path:
	return EVAL_OUT_BASE.joinpath(*parts)


def eval_out_absolute(*parts: str) -> Path:
	return resolve_project_path(eval_out_relative(*parts))



LAMBDA_CLS_FILE = "lambda_values_cls_seed{random_seed}_alpha{alpha}_sources{n_sources}_classes{n_classes}_temperature{temperature}"
LAMBDA_REG_FILE = "lambda_values_reg_seed{random_seed}_alpha{alpha}_sources{n_sources}_temperature{temperature}"

LINEAR_EVAL_FOLDER = eval_out_relative("linear")
LINEAR_EVAL_FOLDER_ABS = eval_out_absolute("linear")
LINEAR_EVAL_LAMBDA_FOLDER = eval_out_relative("linear", "lambda")
LINEAR_EVAL_LAMBDA_FOLDER_ABS = eval_out_absolute("linear", "lambda")

# Backwards-compatible aliases (older code referred to the linear evaluation as
# "iterative_eval"). New code should prefer the LINEAR_EVAL_* names.
ITERATIVE_EVAL_FOLDER = LINEAR_EVAL_FOLDER
ITERATIVE_EVAL_FOLDER_ABS = LINEAR_EVAL_FOLDER_ABS
ITERATIVE_EVAL_LAMBDA_FOLDER = LINEAR_EVAL_LAMBDA_FOLDER
ITERATIVE_EVAL_LAMBDA_FOLDER_ABS = LINEAR_EVAL_LAMBDA_FOLDER_ABS

MDCP_BASELINE_COMP_FIG = "mdcp_vs_baseline_seed{random_seed}_alpha{alpha}_sources{n_sources}_classes{n_classes}_temperature{temperature}"
BASELINE_BASELINE_COMP_FIG_CLS = "baseline_methods_classification_seed{random_seed}_alpha{alpha}_temperature{temperature}"
BASELINE_BASELINE_COMP_FIG_REG = "baseline_methods_regression_seed{random_seed}_alpha{alpha}_temperature{temperature}"

GAMMA_TUNING_FOLDER = eval_out_relative("gamma_tuning")
GAMMA_TUNING_FOLDER_ABS = eval_out_absolute("gamma_tuning")

NONLINEAR_FOLDER = eval_out_relative("nonlinear")
NONLINEAR_FOLDER_ABS = eval_out_absolute("nonlinear")
NONLINEAR_CLASSIFICATION_FOLDER = NONLINEAR_FOLDER / "classification"
NONLINEAR_CLASSIFICATION_FOLDER_ABS = eval_out_absolute("nonlinear", "classification")
NONLINEAR_REGRESSION_FOLDER = NONLINEAR_FOLDER / "regression"
NONLINEAR_REGRESSION_FOLDER_ABS = eval_out_absolute("nonlinear", "regression")
NONLINEAR_LAMBDA_FOLDER = NONLINEAR_FOLDER / "lambda"
NONLINEAR_LAMBDA_FOLDER_ABS = eval_out_absolute("nonlinear", "lambda")

DATA_DENSITY_FOLDER = eval_out_relative("data_density")
DATA_DENSITY_FOLDER_ABS = eval_out_absolute("data_density")


# the following are only for simple run of `evaluation_compact.py`
LAMBDA_FOLDER = eval_out_relative("lambda")
LAMBDA_FOLDER_ABS = eval_out_absolute("lambda")
FIG_FOLDER = eval_out_relative("fig")
FIG_FOLDER_ABS = eval_out_absolute("fig")
MDCP_BASELINE_TAG = "mdcp vs. baseline"
BASELINE_TAG = "baseline"