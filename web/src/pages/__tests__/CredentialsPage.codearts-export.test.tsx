import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CredentialsPage } from "../CredentialsPage";
import {
  expandCredentialGroups, jsonResponse, makeCredential, renderPage, settle, userEvent,
  mockCredentialsFetch,
} from "@/test-utils";

const ADMIN = { username: "root", is_admin: true } as const;
const VIEWER = { username: "guest", is_admin: false } as const;
const ACCOUNT = { user_id: "one", cloud_dragon_token: "token", dpop_private_key: { kty: "EC" } };

afterEach(() => vi.restoreAllMocks());

function setup(isOperator = true, fail = false) {
  const blobs: Blob[] = [];
  const filenames: string[] = [];
  vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
    blobs.push(blob as Blob);
    return "blob:test-export";
  });
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    filenames.push(this.download);
  });
  const spy = mockCredentialsFetch({
    "/api/credentials/codearts/export": fail
      ? () => jsonResponse({ error: { message: "credential not found" } }, 400)
      : [ACCOUNT],
    "/api/credentials": {
      credentials: [
        makeCredential({ id: "ca1", provider: "codearts", nickname: "主号" }),
        makeCredential({ id: "ca2", provider: "codearts", nickname: "备用号" }),
        makeCredential({ id: "tr1", provider: "trae", nickname: "其他渠道" }),
      ],
      viewer: isOperator ? "root" : "guest", is_admin: isOperator, is_operator: isOperator,
    },
  });
  renderPage(<CredentialsPage />, isOperator ? ADMIN : VIEWER);
  const calls = () => spy.mock.calls.filter(([url]) =>
    String(url) === "/api/credentials/codearts/export");
  return { blobs, filenames, revoke, calls };
}

describe("CodeArts 插件格式导出", () => {
  it("勾选导出使用可见的选中项，下载数组 JSON 并保留选择", async () => {
    const result = setup();
    await settle();
    await expandCredentialGroups();
    await userEvent.click(screen.getByTestId("group-select-codearts"));
    await userEvent.type(screen.getByTestId("credential-search"), "主号");
    await userEvent.click(screen.getByTestId("batch-export-codearts"));
    await waitFor(() => expect(result.blobs).toHaveLength(1));
    expect(JSON.parse(String(result.calls()[0][1]?.body))).toEqual({ ids: ["ca1"] });
    expect(result.blobs[0].type).toBe("application/json");
    expect(JSON.parse(await result.blobs[0].text())).toEqual([ACCOUNT]);
    expect(result.filenames[0]).toMatch(/^codearts_1_\d{8}_\d{6}\.json$/);
    expect(result.revoke).toHaveBeenCalledWith("blob:test-export");
    expect(screen.getByTestId("batch-selected-count")).toHaveTextContent("已选 1 个");
    expect(screen.getByTestId("credentials-notice")).toHaveTextContent("已导出 1 个 CodeArts 账号");
  });

  it("渠道导出全部不受筛选影响", async () => {
    const result = setup();
    await settle();
    await userEvent.type(screen.getByTestId("credential-search"), "主号");
    await userEvent.click(screen.getByTestId("export-provider-codearts"));
    await waitFor(() => expect(result.blobs).toHaveLength(1));
    expect(JSON.parse(String(result.calls()[0][1]?.body))).toEqual({});
    expect(result.calls()[0][1]?.method).toBe("POST");
  });

  it("混选其他渠道时隐藏 CodeArts 导出选中", async () => {
    setup();
    await settle();
    await expandCredentialGroups();
    await userEvent.click(screen.getByTestId("select-ca1"));
    expect(screen.getByTestId("batch-export-codearts")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("select-tr1"));
    expect(screen.queryByTestId("batch-export-codearts")).not.toBeInTheDocument();
  });

  it("只读用户没有导出入口", async () => {
    setup(false);
    await settle();
    expect(screen.queryByTestId("export-provider-codearts")).not.toBeInTheDocument();
    expect(screen.queryByTestId("batch-export-codearts")).not.toBeInTheDocument();
  });

  it("导出失败显示原因且不下载文件", async () => {
    const result = setup(true, true);
    await settle();
    await userEvent.click(screen.getByTestId("export-provider-codearts"));
    expect(await screen.findByTestId("credentials-error")).toHaveTextContent("credential not found");
    expect(result.blobs).toHaveLength(0);
    expect(screen.getByTestId("export-provider-codearts")).toBeEnabled();
  });
});
