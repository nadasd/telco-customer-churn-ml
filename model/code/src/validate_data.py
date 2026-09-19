import great_expectations as gx


REQUIRED_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
    "churn",
]


CATEGORICAL_VALUE_SETS = {
    "gender": ["male", "female"],
    "partner": ["yes", "no"],
    "dependents": ["yes", "no"],
    "phoneservice": ["yes", "no"],
    "multiplelines": [
        "yes",
        "no",
        "no phone service",
    ],
    "internetservice": [
        "dsl",
        "fiber optic",
        "no",
    ],
    "onlinesecurity": [
        "yes",
        "no",
        "no internet service",
    ],
    "onlinebackup": [
        "yes",
        "no",
        "no internet service",
    ],
    "deviceprotection": [
        "yes",
        "no",
        "no internet service",
    ],
    "techsupport": [
        "yes",
        "no",
        "no internet service",
    ],
    "streamingtv": [
        "yes",
        "no",
        "no internet service",
    ],
    "streamingmovies": [
        "yes",
        "no",
        "no internet service",
    ],
    "contract": [
        "month-to-month",
        "one year",
        "two year",
    ],
    "paperlessbilling": [
        "yes",
        "no",
    ],
    "paymentmethod": [
        "electronic check",
        "mailed check",
        "bank transfer (automatic)",
        "credit card (automatic)",
    ],
}


def validate_data(df):
    """Validate the cleaned Telco Customer Churn dataset."""

    print("Starting data validation...")

    context = gx.get_context(mode="file")

    # --------------------------------------------------
    # Data source
    # --------------------------------------------------

    try:
        data_source = context.data_sources.get(
            "telco_data_source"
        )
    except Exception:
        data_source = context.data_sources.add_pandas(
            name="telco_data_source"
        )

    # --------------------------------------------------
    # Data asset
    # --------------------------------------------------

    try:
        data_asset = data_source.get_asset(
            "telco_data"
        )
    except Exception:
        data_asset = data_source.add_dataframe_asset(
            name="telco_data"
        )

    # --------------------------------------------------
    # Batch definition
    # --------------------------------------------------

    try:
        batch_definition = (
            data_asset.get_batch_definition(
                "telco_batch"
            )
        )
    except Exception:
        batch_definition = (
            data_asset.add_batch_definition_whole_dataframe(
                "telco_batch"
            )
        )

    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": df
        }
    )

    # --------------------------------------------------
    # Expectations
    # --------------------------------------------------

    expectations = []

    # All expected cleaned columns must exist
    for column in REQUIRED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnToExist(
                column=column
            )
        )

    # No missing values in the canonical cleaned dataset
    for column in REQUIRED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column
            )
        )

    # Allowed categorical values
    for column, values in CATEGORICAL_VALUE_SETS.items():
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=values
            )
        )

    # Senior citizen
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="seniorcitizen",
            value_set=[0, 1]
        )
    )

    # Target
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="churn",
            value_set=[0, 1]
        )
    )

    # Numeric ranges
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure",
            min_value=0,
            max_value=72
        )
    )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="monthlycharges",
            min_value=0
        )
    )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="totalcharges",
            min_value=0
        )
    )

    # --------------------------------------------------
    # Run validation
    # --------------------------------------------------

    failed_expectations = []

    for expectation in expectations:
        result = batch.validate(expectation)

        if not result["success"]:
            failed_expectations.append(
                type(expectation).__name__
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    total_checks = len(expectations)
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks

    print(
        f"Validation: "
        f"{passed_checks}/{total_checks} checks passed."
    )

    if failed_checks == 0:
        print("Data validation PASSED.")
        return True

    print("Data validation FAILED.")
    print("Failed checks:")

    for expectation in failed_expectations:
        print(f"- {expectation}")

    return False