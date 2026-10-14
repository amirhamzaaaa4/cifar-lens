from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_opens_and_class_filter_works():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert app.metric[0].value == '85.07%'
    # When weights exist, ensure the saved error gallery can be filtered.
    if len(app.selectbox) > 1:
        app.selectbox(key='error_class').select('cat').run()
        assert not app.exception
