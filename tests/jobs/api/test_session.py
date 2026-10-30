# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from domyn_swarm.jobs.api.session import new_session_id, session_request_options


def test_session_options_add_header_and_body_field():
    opts = session_request_options("abc", {"temperature": 0.1}, final=False)
    assert opts["extra_headers"] == {"X-Session-ID": "abc"}
    assert opts["extra_body"] == {"temperature": 0.1, "session_params": {"session_id": "abc"}}


def test_session_options_mark_the_final_turn():
    opts = session_request_options("abc", {}, final=True)
    assert opts["extra_headers"]["X-Dynamo-Session-Final"] == "true"


def test_user_session_params_win_and_input_is_not_mutated():
    body = {"session_params": {"session_id": "mine", "x": 1}}
    opts = session_request_options("abc", body, final=False)
    assert opts["extra_body"]["session_params"] == {"session_id": "mine", "x": 1}
    assert body == {"session_params": {"session_id": "mine", "x": 1}}


def test_user_session_id_is_sent_in_the_header_too():
    opts = session_request_options("abc", {"session_params": {"session_id": "mine"}}, final=True)
    assert opts["extra_headers"] == {"X-Session-ID": "mine", "X-Dynamo-Session-Final": "true"}
    assert opts["extra_body"]["session_params"]["session_id"] == "mine"


def test_session_id_is_added_to_user_session_params_without_one():
    body = {"session_params": {"x": 1}}
    opts = session_request_options("abc", body, final=False)
    assert opts["extra_body"]["session_params"] == {"x": 1, "session_id": "abc"}
    assert body == {"session_params": {"x": 1}}


def test_new_session_ids_are_random_hex():
    a, b = new_session_id(), new_session_id()
    assert a != b and len(a) == 32 and int(a, 16) >= 0
