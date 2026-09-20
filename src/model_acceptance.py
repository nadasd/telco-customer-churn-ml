from typing import Dict


def check_model_acceptance(
    metrics: Dict[str, float],
    acceptance_config: Dict[str, float],
) -> dict:
    """
    Apply model promotion rules.

    Rules:
    - precision >= minimum_precision
    - recall >= minimum_recall
    - f1 >= minimum_f1
    - train-test F1 gap <= maximum_f1_train_test_gap
    """

    checks = {
        "precision": (
            metrics["precision"]
            >= acceptance_config["minimum_precision"]
        ),
        "recall": (
            metrics["recall"]
            >= acceptance_config["minimum_recall"]
        ),
        "f1": (
            metrics["f1"]
            >= acceptance_config["minimum_f1"]
        ),
        "generalization": (
            metrics["f1_gap"]
            <= acceptance_config["maximum_f1_train_test_gap"]
        ),
    }

    return {
        "accepted": all(checks.values()),
        "checks": checks,
    }


def print_acceptance_report(result: dict):
    print("\nModel Acceptance Report")
    print("-----------------------")

    for name, passed in result["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    if result["accepted"]:
        print("\nMODEL ACCEPTED")
    else:
        print("\nMODEL REJECTED")