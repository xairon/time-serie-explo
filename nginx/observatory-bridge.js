/**
 * Observatory Bridge Script
 * Injected via nginx sub_filter into the junondashboard iframe.
 * Detects station selection and sends postMessage to the parent Junon Explorer.
 */
(function () {
  if (window.parent === window) return; // Not in iframe, skip

  var lastCode = null;

  function extractStationInfo() {
    var drawer = document.querySelector('[role="dialog"]');
    if (!drawer) return null;

    var ariaLabel = drawer.getAttribute('aria-label') || '';
    var match = ariaLabel.match(/Station\s+(.+)/);
    if (!match) return null;

    var code = match[1].trim();

    // Detect type from badge
    var typeBadge = drawer.querySelector('[class*="bg-accent-cyan"]');
    var type = 'piezo';
    if (!typeBadge) {
      typeBadge = drawer.querySelector('[class*="bg-accent-indigo"]');
      if (typeBadge) type = 'hydro';
    }

    // Extract station name from h3
    var h3 = drawer.querySelector('h3');
    var name = h3 ? h3.textContent.trim() : code;

    // Extract department from secondary text
    var secondary = drawer.querySelector('.text-xs.text-text-secondary');
    var dept = '';
    if (secondary) {
      var parts = secondary.textContent.split('\u00b7');
      if (parts.length > 0) dept = parts[0].trim();
    }

    return { code: code, type: type, name: name, dept: dept };
  }

  function addTrainButton(drawer, info) {
    if (drawer.querySelector('.junon-train-btn')) return;

    var btn = document.createElement('button');
    btn.className = 'junon-train-btn';
    btn.textContent = '\u26A1 Analyser dans Junon';
    btn.style.cssText =
      'display:block;width:calc(100% - 32px);margin:12px 16px;padding:10px 16px;' +
      'background:linear-gradient(135deg, #06b6d4, #3b82f6);color:#fff;border:none;' +
      'border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;' +
      'transition:opacity 0.2s;text-align:center;';
    btn.onmouseenter = function () { btn.style.opacity = '0.85'; };
    btn.onmouseleave = function () { btn.style.opacity = '1'; };
    btn.onclick = function () {
      window.parent.postMessage(
        { type: 'OBSERVATORY_TRAIN', station: info },
        '*'
      );
    };

    // Insert after the header section (first border-t or after first few children)
    var sections = drawer.querySelectorAll('.border-t');
    if (sections.length > 0) {
      sections[0].parentNode.insertBefore(btn, sections[0]);
    } else {
      drawer.appendChild(btn);
    }
  }

  function check() {
    var info = extractStationInfo();
    if (info && info.code !== lastCode) {
      lastCode = info.code;
      // Notify parent of selection
      window.parent.postMessage(
        { type: 'OBSERVATORY_STATION_SELECTED', station: info },
        '*'
      );
    }
    if (!info) {
      if (lastCode !== null) {
        lastCode = null;
        window.parent.postMessage(
          { type: 'OBSERVATORY_STATION_DESELECTED' },
          '*'
        );
      }
    }

    // Add train button if drawer is open
    var drawer = document.querySelector('[role="dialog"]');
    if (drawer && info) {
      addTrainButton(drawer, info);
    }
  }

  // Use MutationObserver for efficiency
  var observer = new MutationObserver(function () {
    check();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // Also check periodically as fallback
  setInterval(check, 1000);
})();
