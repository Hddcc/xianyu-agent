"""路由测试。全部不调真实模型：规则命中、正则命中、兜底分类、no_reply。

路由是最不该出 bug 的地方——把议价消息回成一张参数表，买家当场就走了。
"""


def make_router(classify_result="default"):
    calls = []

    def classify(user_msg, item_desc, history):
        calls.append(user_msg)
        return classify_result

    from xianyu_agent.router import IntentRouter
    return IntentRouter(classify), calls


def test_price_keyword():
    router, calls = make_router()
    assert router.detect("能便宜点吗", "", "") == "price"
    assert calls == []                     # 规则命中就不问模型


def test_price_pattern():
    router, _ = make_router()
    assert router.detect("300元卖不卖", "", "") == "price"
    assert router.detect("能少50吗", "", "") == "price"


def test_tech_pattern():
    router, calls = make_router()
    assert router.detect("和iphone16比哪个好", "", "") == "tech"
    assert calls == []


def test_tech_keyword():
    router, _ = make_router()
    assert router.detect("这个规格是什么样的", "", "") == "tech"
    assert router.detect("支持什么型号", "", "") == "tech"


def test_tech_has_priority_over_price():
    # 技术优先：既有参数又有价格词时归 tech（原项目经验）
    router, _ = make_router()
    assert router.detect("这个型号多少钱", "", "") == "tech"


def test_fallback_classify_default():
    router, calls = make_router("default")
    assert router.detect("在吗", "", "") == "default"
    assert calls == ["在吗"]               # 规则没命中，问了模型


def test_fallback_classify_no_reply():
    router, _ = make_router("no_reply")
    assert router.detect("你是AI吗", "", "") == "no_reply"


def test_fallback_classify_price():
    router, _ = make_router("price")
    assert router.detect("最低多少出", "", "") == "price"
