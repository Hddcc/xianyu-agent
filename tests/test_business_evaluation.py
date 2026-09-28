from benchmarks.business_scenarios import detail_contains_amount


def test_amount_comparison_treats_equivalent_numeric_formats_as_equal():
    assert detail_contains_amount("买家确认 348.0 元，请手动改价", "348")
    assert detail_contains_amount("买家要求包邮", "0")
    assert not detail_contains_amount("买家确认 348 元", "0")
