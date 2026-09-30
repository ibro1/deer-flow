import { expect, test, type Route } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

// A message sent while a response is streaming waits in a queue above the
// composer and goes out as the next run once the current one finishes.
test.describe("Sending while a response streams", () => {
  test("queues the message and sends it when the run finishes", async ({
    page,
  }) => {
    mockLangGraphAPI(page);
    const runBodies: string[] = [];
    // Registered after the mock, so it runs first: record each run request and
    // hold the first one open so the thread is visibly streaming.
    const holdFirstRun = async (route: Route) => {
      runBodies.push(route.request().postData() ?? "");
      if (runBodies.length === 1) {
        await new Promise((resolve) => setTimeout(resolve, 3_000));
      }
      await route.fallback();
    };
    await page.route("**/api/langgraph/runs/stream", holdFirstRun);
    await page.route("**/api/langgraph/threads/*/runs/stream", holdFirstRun);

    await page.goto("/workspace/chats/new");
    const textarea = page.getByPlaceholder(/how can i assist you/i);
    await expect(textarea).toBeVisible({ timeout: 15_000 });

    await textarea.fill("First question");
    await textarea.press("Enter");
    await expect.poll(() => runBodies.length).toBe(1);

    const composer = page.locator("textarea").last();
    await composer.fill("Follow-up sent mid-run");
    await composer.press("Enter");

    await expect(
      page.getByText("Queued: sends when this response finishes"),
    ).toBeVisible();
    await expect(
      page.getByText("Follow-up sent mid-run", { exact: true }),
    ).toBeVisible();
    await expect(composer).toHaveValue("");

    // The first run ends; the queued message becomes the second run.
    await expect.poll(() => runBodies.length, { timeout: 15_000 }).toBe(2);
    expect(runBodies[1]).toContain("Follow-up sent mid-run");
    await expect(
      page.getByText("Queued: sends when this response finishes"),
    ).toBeHidden();
  });

  test("a removed message is never sent", async ({ page }) => {
    mockLangGraphAPI(page);
    const runBodies: string[] = [];
    const holdFirstRun = async (route: Route) => {
      runBodies.push(route.request().postData() ?? "");
      if (runBodies.length === 1) {
        await new Promise((resolve) => setTimeout(resolve, 3_000));
      }
      await route.fallback();
    };
    await page.route("**/api/langgraph/runs/stream", holdFirstRun);
    await page.route("**/api/langgraph/threads/*/runs/stream", holdFirstRun);

    await page.goto("/workspace/chats/new");
    const textarea = page.getByPlaceholder(/how can i assist you/i);
    await expect(textarea).toBeVisible({ timeout: 15_000 });
    await textarea.fill("First question");
    await textarea.press("Enter");
    await expect.poll(() => runBodies.length).toBe(1);

    const composer = page.locator("textarea").last();
    await composer.fill("Never mind this one");
    await composer.press("Enter");
    await page.getByRole("button", { name: "Remove from queue" }).click();
    await expect(
      page.getByText("Never mind this one", { exact: true }),
    ).toBeHidden();

    await page.waitForTimeout(5_000);
    expect(runBodies).toHaveLength(1);
  });
});
