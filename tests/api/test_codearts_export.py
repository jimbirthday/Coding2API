from __future__ import annotations

import re
import time

import pytest

from src.auth.session import create_session_token
from src.provider.codearts import CodeArtsProvider
from src.provider.codearts.credential import CodeArtsCredential
from tests.helpers.constants import SECRET
from tests.provider.codearts._support import jwk


def test_codearts_export_matches_plugin_format_and_roundtrips(admin_client):
    app, client = admin_client
    private = jwk()
    data = CodeArtsProvider().import_credential({
        "user_id": "test-user", "user_name": "test-name", "domain_id": "test-domain",
        "cloud_dragon_token": "test-security", "access_key_id": "test-ak",
        "secret_access_key": "test-sk", "expiration": "2026-11-21T04:30:00.515Z",
        "refresh_token": "test-rt", "code_verifier": "test-verifier",
        "dpop_private_key": private, "web_cookie": "private-browser-cookie",
    })
    credential_id = app.state.credentials.add(provider="codearts", credential_data=data)
    before = int(time.time())
    response = client.post("/api/credentials/codearts/export", json={"ids": [credential_id]})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert re.fullmatch(r'attachment; filename="codearts_1_\d{8}_\d{6}\.json"',
                        response.headers["content-disposition"])
    account, = response.json()
    assert account == {
        "user_id": "test-user", "user_name": "test-name", "domain_id": "test-domain",
        "cloud_dragon_token": "test-security", "access_key_id": "test-ak",
        "secret_access_key": "test-sk", "expiration": "2026-11-21T04:30:00.000Z",
        "refresh_token": "test-rt", "code_verifier": "test-verifier",
        "updated_at": account["updated_at"], "client_id": "codearts-agent",
        "dpop_private_key": private,
    }
    assert before <= account["updated_at"] <= int(time.time())
    restored = CodeArtsProvider().import_credential(account)
    assert restored == {**data, "web_cookie": ""}
    assert app.state.credentials.credential_data(credential_id) == data
    audit, = app.state.audit.query(action="credential.export")
    assert audit["actor"] == "root" and audit["target"] is None
    assert audit["detail"] == "CodeArts 插件格式：导出 1 个账号"
    assert "test-rt" not in str(audit)


def test_export_all_includes_paused_and_disabled_and_excludes_other_providers(admin_client):
    app, client = admin_client
    first = app.state.credentials.add(provider="codearts", credential_data={"uid": "one"})
    second = app.state.credentials.add(provider="codearts", credential_data={"uid": "two"})
    app.state.credentials.add(provider="trae", credential_data={"accessToken": "do-not-export"})
    app.state.credentials.set_enabled(first, False)
    app.state.credentials.mark_relogin_required(second, "expired")
    response = client.post("/api/credentials/codearts/export", json={})
    assert response.status_code == 200
    assert {row["user_id"] for row in response.json()} == {"one", "two"}
    assert all(row["expiration"] == "" for row in response.json())
    assert "do-not-export" not in response.text


def test_export_selected_preserves_order_deduplicates_and_excludes_unselected(admin_client):
    app, client = admin_client
    ids = [app.state.credentials.add(provider="codearts", credential_data={"uid": str(index)})
           for index in range(3)]
    response = client.post("/api/credentials/codearts/export",
                           json={"ids": [ids[1], f" {ids[0]} ", ids[1]]})
    assert response.status_code == 200
    assert [item["user_id"] for item in response.json()] == ["1", "0"]


@pytest.mark.parametrize("ids", [None, [], "one", [None], [1], ["  "], [{}], ["x"] * 201])
def test_export_rejects_invalid_selection(admin_client, ids):
    app, client = admin_client
    assert client.post("/api/credentials/codearts/export", json={"ids": ids}).status_code == 400
    assert not app.state.audit.query(action="credential.export")


def test_export_rejects_missing_or_wrong_provider_without_partial_file(admin_client):
    app, client = admin_client
    codearts = app.state.credentials.add(provider="codearts", credential_data={"uid": "one"})
    other = app.state.credentials.add(provider="trae", credential_data={})
    for bad_id in ("missing", other):
        response = client.post("/api/credentials/codearts/export", json={"ids": [codearts, bad_id]})
        assert response.status_code == 400
        assert "cloud_dragon_token" not in response.text
    assert not app.state.audit.query(action="credential.export")


def test_empty_export_and_permissions(admin_client):
    app, client = admin_client
    assert client.post("/api/credentials/codearts/export", json={}).json() == []
    rejected = client.post("/api/credentials/codearts/export", json={},
                           headers={"Origin": "https://evil.example"})
    assert rejected.status_code == 403
    client.cookies.set("coding2api_session", create_session_token("guest", SECRET))
    assert client.post("/api/credentials/codearts/export", json={}).status_code == 403
    app.state.services.user_repo.update_role("alice", "operator", bump_epoch=False)
    client.cookies.set("coding2api_session", create_session_token("alice", SECRET))
    assert client.post("/api/credentials/codearts/export", json={}).status_code == 200


def test_plugin_dict_keeps_default_client_id_and_missing_fields():
    exported = CodeArtsCredential.from_dict({}).to_plugin_dict(updated_at=100)
    assert exported["expiration"] == "" and exported["updated_at"] == 100
    assert exported["dpop_private_key"] == {} and exported["client_id"] == "codearts-agent"


def test_export_one_hundred_accounts_as_single_json_array(admin_client):
    app, client = admin_client
    ids = [app.state.credentials.add(provider="codearts", credential_data={"uid": str(index)})
           for index in range(100)]
    response = client.post("/api/credentials/codearts/export", json={"ids": ids})
    assert response.status_code == 200
    assert len(response.json()) == 100
    assert [row["user_id"] for row in response.json()] == [str(index) for index in range(100)]
    assert 'filename="codearts_100_' in response.headers["content-disposition"]
