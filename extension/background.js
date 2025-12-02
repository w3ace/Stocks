const TARGET_URL = "https://www.ebay.com/sh/lst/active";
const EBAY_HOSTS = new Set(["www.ebay.com", "ebay.com"]);

function isRootEbayUrl(urlString) {
  try {
    const url = new URL(urlString);
    const isEbayHost = EBAY_HOSTS.has(url.hostname.toLowerCase());

    if (!isEbayHost) {
      return false;
    }

    // Normalize trailing slashes so "/" and "//" both collapse to "/".
    const normalizedPath = url.pathname.replace(/\/+$/, "/");
    return normalizedPath === "/";
  } catch (error) {
    return false;
  }
}

chrome.webNavigation.onCommitted.addListener((details) => {
  if (details.frameId !== 0) {
    return; // Only redirect top-level navigations.
  }

  if (!isRootEbayUrl(details.url)) {
    return;
  }

  if (details.url === TARGET_URL) {
    return; // Already on the destination.
  }

  chrome.tabs.update(details.tabId, { url: TARGET_URL });
});
