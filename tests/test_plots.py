from ffonons.plots import plotly_title


def test_plotly_title() -> None:
    assert plotly_title("Si2") == "Si<sub>2</sub>"

    mp_id_title = 'Si<sub>2</sub>  <a href="https://legacy.materialsproject.org/materials/mp-149">mp-149</a>'
    assert plotly_title("Si2", "mp-149") == mp_id_title

    random_url_title = (
        'Fe<sub>2</sub>O<sub>3</sub>  <a href="https://example.com">example.com</a>'
    )
    assert plotly_title("Fe2O3", "https://example.com") == random_url_title
