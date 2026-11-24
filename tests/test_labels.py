import pytest

from opf_mlx.labels import LabelError, resolve_label_space


def test_released_label_space_is_bioes_over_eight_categories(released_config):
    space = resolve_label_space(released_config)

    assert len(space.class_names) == 33
    assert space.class_names[0] == "O"
    assert space.class_names[1:5] == (
        "B-account_number",
        "I-account_number",
        "E-account_number",
        "S-account_number",
    )
    assert space.class_names[20] == "S-private_person"
    assert space.class_names[32] == "S-secret"
    assert space.span_names == (
        "O",
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    )
    assert space.category_version == "v2"


def test_each_token_class_maps_to_its_span_class_and_boundary_tag(released_config):
    space = resolve_label_space(released_config)

    assert space.background == 0
    assert (space.span_of[0], space.tag_of[0]) == (0, None)
    assert (space.span_of[17], space.tag_of[17]) == (5, "B")
    assert (space.span_of[18], space.tag_of[18]) == (5, "I")
    assert (space.span_of[19], space.tag_of[19]) == (5, "E")
    assert (space.span_of[20], space.tag_of[20]) == (5, "S")
    assert (space.span_of[32], space.tag_of[32]) == (8, "S")


def test_unknown_label_count_is_rejected(released_config):
    released_config["num_labels"] = 34

    with pytest.raises(LabelError, match="num_labels=34"):
        resolve_label_space(released_config)


def test_category_version_that_contradicts_num_labels_is_rejected(released_config):
    released_config["category_version"] = "v4"

    with pytest.raises(LabelError, match="inconsistent"):
        resolve_label_space(released_config)


def test_custom_span_names_expand_in_bies_order_with_background_first(released_config):
    released_config["span_class_names"] = ["pet_name", "O", "badge_id"]
    released_config["num_labels"] = 9

    space = resolve_label_space(released_config)

    assert space.span_names == ("O", "pet_name", "badge_id")
    assert space.class_names == (
        "O",
        "B-pet_name",
        "I-pet_name",
        "E-pet_name",
        "S-pet_name",
        "B-badge_id",
        "I-badge_id",
        "E-badge_id",
        "S-badge_id",
    )
    assert space.category_version == "custom"


def test_custom_token_names_keep_their_order_even_if_background_is_not_first(released_config):
    released_config["ner_class_names"] = ["B-x", "I-x", "O", "E-x", "S-x"]
    released_config["num_labels"] = 5

    space = resolve_label_space(released_config)

    assert space.background == 2
    assert space.span_names == ("O", "x")
    assert space.span_of == (1, 1, 0, 1, 1)
    assert space.tag_of == ("B", "I", None, "E", "S")


def test_custom_token_names_missing_a_boundary_are_rejected(released_config):
    released_config["ner_class_names"] = ["O", "B-x", "I-x", "E-x"]
    released_config["num_labels"] = 4

    with pytest.raises(LabelError, match="missing boundaries"):
        resolve_label_space(released_config)


def test_custom_names_must_match_num_labels(released_config):
    released_config["span_class_names"] = ["O", "x"]

    with pytest.raises(LabelError, match="num_labels"):
        resolve_label_space(released_config)


def test_custom_token_names_must_match_the_span_names_expansion(released_config):
    released_config["span_class_names"] = ["O", "x"]
    released_config["ner_class_names"] = ["O", "S-x", "B-x", "I-x", "E-x"]
    released_config["num_labels"] = 5

    with pytest.raises(LabelError, match="do not match"):
        resolve_label_space(released_config)


@pytest.mark.parametrize("names", [["O", "x"], ["O", "Q-x", "B-x", "I-x", "E-x", "S-x"], ["B-x", "I-x", "E-x", "S-x"]])
def test_malformed_custom_token_names_are_rejected(released_config, names):
    released_config["ner_class_names"] = names
    released_config["num_labels"] = len(names)

    with pytest.raises(LabelError):
        resolve_label_space(released_config)
