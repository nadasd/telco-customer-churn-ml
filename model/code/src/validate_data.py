import great_expectations as gx


def validate_data(df):
    """Validate the cleaned Telco Customer Churn dataset."""

    print("Starting data validation...")

    # Create Great Expectations context
    context = gx.get_context(mode="file")

    # Get existing datasource or create it
    try:
        data_source = context.data_sources.get("telco_data_source")
    except Exception:
        data_source = context.data_sources.add_pandas(
            name="telco_data_source"
        )

    # Get existing data asset or create it
    try:
        data_asset = data_source.get_asset("telco_data")
    except Exception:
        data_asset = data_source.add_dataframe_asset(
            name="telco_data"
        )

    # Get existing batch definition or create it
    try:
        batch_definition = data_asset.get_batch_definition(
            "telco_batch"
        )
    except Exception:
        batch_definition = (
            data_asset.add_batch_definition_whole_dataframe(
                "telco_batch"
            )
        )

    # Create batch from DataFrame
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    # -----------------------------
    # Expectations
    # -----------------------------

    expectations = [

        # Required columns
        gx.expectations.ExpectColumnToExist(
            column="gender"
        ),

        gx.expectations.ExpectColumnToExist(
            column="seniorcitizen"
        ),

        gx.expectations.ExpectColumnToExist(
            column="partner"
        ),

        gx.expectations.ExpectColumnToExist(
            column="dependents"
        ),

        gx.expectations.ExpectColumnToExist(
            column="tenure"
        ),

        gx.expectations.ExpectColumnToExist(
            column="monthlycharges"
        ),

        gx.expectations.ExpectColumnToExist(
            column="totalcharges"
        ),

        gx.expectations.ExpectColumnToExist(
            column="churn"
        ),

        # No missing values
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="churn"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="tenure"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="monthlycharges"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="totalcharges"
        ),

        # Categorical values
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender",
            value_set=["male", "female"]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="partner",
            value_set=["yes", "no"]
        ),

        gx.expectations.ExpectColumnValuesToBeInSet(
            column="dependents",
            value_set=["yes", "no"]
        ),

        # Target
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="churn",
            value_set=[0, 1]
        ),

        # Numeric ranges
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure",
            min_value=0,
            max_value=72
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="monthlycharges",
            min_value=0
        ),

        gx.expectations.ExpectColumnValuesToBeBetween(
            column="totalcharges",
            min_value=0
        ),
    ]

    # -----------------------------
    # Run validation
    # -----------------------------

    failed_expectations = []

    for expectation in expectations:

        result = batch.validate(expectation)

        if not result["success"]:
            failed_expectations.append(
                expectation.expectation_type
            )

    # -----------------------------
    # Results
    # -----------------------------

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