(function () {
  "use strict";

  var dataNode = document.getElementById("report-data");
  if (!dataNode) {
    return;
  }

  var payload;
  try {
    payload = JSON.parse(dataNode.textContent || "{}");
  } catch (error) {
    document.getElementById("report-root").textContent =
      "Yewo could not read the saved report data.";
    return;
  }

  var record = payload.record;
  applySavedTheme();
  populateHeader(record, payload.report);
  populateVerdict(record);
  populateTiles(record);
  populateTestFlow(record, payload.report.title);
  populateIssues(record);
  populateScenarios(record);
  connectThemeButton();

  function populateHeader(runRecord, report) {
    setText("generated-at", formatTimestamp(runRecord.completed_at));
    setText("yewo-version", report.yewo_version);
    setText("journey-id", journeyId(runRecord));
    setText("journey-title", report.title);
  }

  function populateVerdict(runRecord) {
    var tone = verdictTone(runRecord.qa_verdict);
    var passedCount = runRecord.scenarios.filter(function (scenario) {
      return scenario.audit.verdict === "passed";
    }).length;
    var verdict = document.getElementById("verdict");
    var badge = document.getElementById("verdict-badge");
    verdict.className = "verdict t-" + tone;
    badge.className = "badge b-" + tone;
    badge.textContent = titleCase(runRecord.qa_verdict);
    setText(
      "verdict-count",
      passedCount + " of " + runRecord.scenarios.length + " scenarios passed"
    );

    var problem = firstProblem(runRecord);
    var link = document.getElementById("verdict-link");
    if (problem) {
      setText(
        "verdict-headline",
        "“" + problem.scenario.name + "” " + problem.scenario.audit.verdict + "."
      );
      setText("verdict-detail", problem.scenario.audit.summary);
      link.href = "#" + scenarioAnchor(problem.scenario);
      link.hidden = false;
      return;
    }
    setText("verdict-headline", runRecord.terminal_reason);
    setText(
      "verdict-detail",
      runRecord.scenarios.length
        ? "Every requested scenario passed."
        : "The run has no executable scenarios."
    );
  }

  function populateTiles(runRecord) {
    var scenarioCount = runRecord.scenarios.length;
    var passedCount = runRecord.scenarios.filter(function (scenario) {
      return scenario.audit.verdict === "passed";
    }).length;
    var stepCount = runRecord.scenarios.reduce(function (total, scenario) {
      return total + scenario.interactions.length;
    }, 0);
    var durationSeconds =
      (new Date(runRecord.completed_at) - new Date(runRecord.started_at)) / 1000;
    var callCount = runRecord.usage.model_call_count;

    setText("tile-scenarios", passedCount + " of " + scenarioCount + " passed");
    setText("tile-steps", String(stepCount));
    setText("tile-steps-note", "across " + scenarioCount + " scenarios");
    setText("tile-duration", formatDuration(durationSeconds));
    setText(
      "tile-cost",
      callCount ? "$" + Number(runRecord.usage.known_cost_usd).toFixed(2) : "Not recorded"
    );
    setText(
      "tile-cost-note",
      callCount ? callCount + " model calls" : "no usage telemetry"
    );
    setText("tile-model", modelName(runRecord));
    setText("tile-device", deviceName(runRecord));
    setText("tile-build", applicationName(runRecord));
  }

  function populateTestFlow(runRecord, journeyTitle) {
    var interactionCount = runRecord.scenarios.reduce(function (maximum, scenario) {
      return Math.max(maximum, scenario.interactions.length);
    }, 0);
    setText(
      "flow-summary",
      journeyTitle +
        (interactionCount ? " · " + interactionCount + " recorded interactions" : "")
    );
    setText(
      "flow-description",
      runRecord.scenarios.length
        ? runRecord.scenarios[0].objective
        : runRecord.terminal_reason
    );
  }

  function populateIssues(runRecord) {
    var issues = [];
    runRecord.scenarios.forEach(function (scenario) {
      scenario.audit.obligations.forEach(function (obligation) {
        if (obligation.status === "unmet") {
          issues.push({scenario: scenario, obligation: obligation});
        }
      });
    });
    if (!issues.length) {
      return;
    }
    setText(
      "issue-summary",
      issues.length + " failed check(s) established by current-run evidence"
    );
    var issueList = document.getElementById("issue-list");
    issues.forEach(function (issue) {
      appendIssueCards(runRecord, issue.scenario, issue.obligation, issueList);
    });
    document.getElementById("findings").hidden = false;
  }

  function appendIssueCards(runRecord, scenario, obligation, container) {
    if (!obligation.findings.length) {
      container.appendChild(buildIssueCard(runRecord, scenario, obligation, null));
      return;
    }
    obligation.findings.forEach(function (finding) {
      container.appendChild(buildIssueCard(runRecord, scenario, obligation, finding));
    });
  }

  function buildIssueCard(runRecord, scenario, obligation, finding) {
    var fragment = cloneTemplate("issue-template");
    var card = fragment.querySelector(".issue-card");
    var evidenceContainer = fragment.querySelector(".issue-evidence");
    var copy = fragment.querySelector(".issue-copy");
    copy.querySelector(".rule-label").textContent = obligation.description;
    var heading = copy.querySelector("h3");
    var reason = copy.querySelector("p");
    var link = copy.querySelector("a");

    if (!finding) {
      evidenceContainer.hidden = true;
      heading.textContent = obligation.reason;
      reason.hidden = true;
      link.textContent = "View scenario evidence";
      link.href = "#" + scenarioAnchor(scenario);
      return fragment;
    }

    card.id = findingAnchor(scenario, finding.finding_id);
    heading.textContent = finding.claim;
    reason.textContent = finding.reason;
    link.textContent = "View exact scenario state";
    link.href = "#" + findingStepAnchor(scenario, finding);
    var evidenceIds = new Set();
    finding.regions.forEach(function (region) {
      var evidenceId = region.screenshot_evidence_id;
      if (!evidenceId || evidenceIds.has(evidenceId)) {
        return;
      }
      var evidence = findEvidence(runRecord, evidenceId);
      if (!evidence) {
        return;
      }
      evidenceIds.add(evidenceId);
      evidenceContainer.appendChild(buildIssueScreenshot(evidence, region.label));
    });
    evidenceContainer.hidden = !evidenceContainer.children.length;
    return fragment;
  }

  function buildIssueScreenshot(evidence, label) {
    var fragment = cloneTemplate("issue-shot-template");
    var href = evidenceHref(evidence.path);
    var link = fragment.querySelector("a");
    var image = fragment.querySelector("img");
    link.href = href;
    image.src = href;
    image.alt = label;
    fragment.querySelector("figcaption").textContent = label;
    return fragment;
  }

  function populateScenarios(runRecord) {
    var scenarioList = document.getElementById("scenario-list");
    if (!runRecord.scenarios.length) {
      var emptyFragment = cloneTemplate("empty-scenarios-template");
      emptyFragment.querySelector("p").textContent = runRecord.terminal_reason;
      scenarioList.appendChild(emptyFragment);
      return;
    }
    if (runRecord.scenarios.length > 1) {
      populateScenarioRail(runRecord.scenarios);
    }
    runRecord.scenarios.forEach(function (scenario) {
      scenarioList.appendChild(buildScenario(runRecord, scenario));
    });
  }

  function populateScenarioRail(scenarios) {
    var links = document.getElementById("scenario-links");
    scenarios.forEach(function (scenario) {
      var fragment = cloneTemplate("scenario-link-template");
      var link = fragment.querySelector("a");
      link.href = "#" + scenarioAnchor(scenario);
      var dot = fragment.querySelector(".dot");
      dot.className = "dot d-" + verdictTone(scenario.audit.verdict);
      fragment.querySelector(".scenario-title").textContent = scenario.name;
      fragment.querySelector(".count").textContent =
        titleCase(scenario.audit.verdict) + " · " + scenario.interactions.length + " steps";
      links.appendChild(fragment);
    });
    document.getElementById("scenario-rail").hidden = false;
  }

  function buildScenario(runRecord, scenario) {
    var fragment = cloneTemplate("scenario-template");
    var section = fragment.querySelector(".walk");
    var tone = verdictTone(scenario.audit.verdict);
    section.id = scenarioAnchor(scenario);
    section.querySelector("h2").textContent = scenario.name;
    section.querySelector(".scenario-status").textContent =
      titleCase(scenario.audit.verdict) + " · " + scenario.interactions.length + " steps";
    section.querySelector(".scenario-identity").textContent =
      scenario.coverage_cell_id + " · " + scenario.scenario_id;
    var summary = section.querySelector(".scenario-summary");
    summary.className = "failure-reason scenario-summary t-" + tone;
    summary.textContent = scenario.audit.summary;
    if (scenario.execution.restoration_failure_reason) {
      var warning = section.querySelector(".restoration-warning");
      warning.textContent =
        "Cleanup warning: " + scenario.execution.restoration_failure_reason;
      warning.hidden = false;
    }
    populateInteractions(runRecord, scenario, section);
    populateVerification(scenario, section);
    return fragment;
  }

  function populateInteractions(runRecord, scenario, section) {
    var interactionList = section.querySelector(".interaction-list");
    if (!scenario.interactions.length) {
      section.querySelector(".no-interactions").hidden = false;
      return;
    }
    scenario.interactions.forEach(function (interaction, index) {
      interactionList.appendChild(
        buildInteraction(runRecord, scenario, interaction, index + 1)
      );
    });
  }

  function buildInteraction(runRecord, scenario, interaction, index) {
    var fragment = cloneTemplate("interaction-template");
    var step = fragment.querySelector(".step");
    var tone = interaction.outcome_disposition ? "pass" : "muted";
    var result = interaction.outcome_disposition || interaction.dispatch_disposition;
    var target = interaction.target || "visible target";
    step.id = scenarioAnchor(scenario) + "-step-" + index;
    step.className = "step s-" + tone;
    step.querySelector(".step-number").textContent = String(index).padStart(2, "0");
    step.querySelector(".mark").style.stroke =
      "var(--" + (tone === "pass" ? "pass-text" : "line-strong") + ")";
    step.querySelector(".step-title").textContent =
      titleCase(interaction.action) + " " + target;
    var chip = step.querySelector(".chip");
    chip.className = "chip c-" + tone;
    chip.textContent = result;
    step.querySelector(".expected-outcome").textContent = interaction.expected_outcome;
    step.querySelector(".dispatch-result").textContent = result;
    step.querySelector(".target").textContent = target;
    populateInteractionScreenshot(runRecord, interaction, step);
    return fragment;
  }

  function populateInteractionScreenshot(runRecord, interaction, step) {
    var evidence = findEvidence(runRecord, interaction.screenshot_evidence_id);
    var shots = step.querySelector(".shots");
    if (!evidence) {
      shots.hidden = true;
      step.querySelector(".screenshot-unavailable").hidden = false;
      return;
    }
    var href = evidenceHref(evidence.path);
    var link = shots.querySelector("a");
    var image = shots.querySelector("img");
    link.href = href;
    image.src = href;
    image.alt = "Source screen before " + interaction.action;
    var box = shots.querySelector(".box");
    if (interaction.region) {
      box.style.left = percent(interaction.region.left, interaction.screenshot_width_px);
      box.style.top = percent(interaction.region.top, interaction.screenshot_height_px);
      box.style.width = percent(
        interaction.region.right - interaction.region.left,
        interaction.screenshot_width_px
      );
      box.style.height = percent(
        interaction.region.bottom - interaction.region.top,
        interaction.screenshot_height_px
      );
      box.hidden = false;
    }
    var tap = shots.querySelector(".tap");
    tap.style.left = percent(interaction.point.x, interaction.screenshot_width_px);
    tap.style.top = percent(interaction.point.y, interaction.screenshot_height_px);
    shots.querySelector("figcaption").textContent =
      interaction.action + " · (" + interaction.point.x + ", " + interaction.point.y +
      ") · " + interaction.geometry_source;
  }

  function populateVerification(scenario, section) {
    var unresolved = section.querySelector(".unresolved-checks");
    var passed = section.querySelector(".passed-checks");
    var passedList = section.querySelector(".passed-check-list");
    var passedCount = 0;
    scenario.audit.obligations.forEach(function (obligation) {
      var check = buildCheck(scenario, obligation);
      if (obligation.status === "satisfied") {
        passedList.appendChild(check);
        passedCount += 1;
      } else {
        unresolved.appendChild(check);
      }
    });
    if (passedCount) {
      passed.querySelector("summary").textContent =
        "Show " + passedCount + " passed check(s)";
      passed.hidden = false;
    }
  }

  function buildCheck(scenario, obligation) {
    var fragment = cloneTemplate("check-template");
    var badge = fragment.querySelector(".badge");
    var labels = {satisfied: "Passed", unmet: "Failed", unverified: "Not evidenced"};
    badge.className = "badge b-" + obligationTone(obligation.status);
    badge.textContent = labels[obligation.status];
    fragment.querySelector(".rule").textContent = obligation.description;
    var reasoning = fragment.querySelector(".check-reasoning");
    if (!obligation.findings.length) {
      reasoning.textContent = obligation.reason;
      return fragment;
    }
    reasoning.appendChild(
      document.createTextNode("Established by current-run visual evidence. ")
    );
    obligation.findings.forEach(function (finding, index) {
      if (index) {
        reasoning.appendChild(document.createTextNode(" · "));
      }
      var link = document.createElement("a");
      link.href = "#" + findingAnchor(scenario, finding.finding_id);
      link.textContent = "View issue evidence";
      reasoning.appendChild(link);
    });
    return fragment;
  }

  function firstProblem(runRecord) {
    var statuses = ["unmet", "unverified"];
    for (var statusIndex = 0; statusIndex < statuses.length; statusIndex += 1) {
      for (var scenarioIndex = 0; scenarioIndex < runRecord.scenarios.length;
        scenarioIndex += 1) {
        var scenario = runRecord.scenarios[scenarioIndex];
        for (var obligationIndex = 0;
          obligationIndex < scenario.audit.obligations.length;
          obligationIndex += 1) {
          if (scenario.audit.obligations[obligationIndex].status === statuses[statusIndex]) {
            return {scenario: scenario, obligation: scenario.audit.obligations[obligationIndex]};
          }
        }
      }
    }
    return null;
  }

  function findingStepAnchor(scenario, finding) {
    var observationIds = new Set(finding.regions.map(function (region) {
      return region.observation_id;
    }));
    for (var index = 0; index < scenario.interactions.length; index += 1) {
      var interaction = scenario.interactions[index];
      if (
        observationIds.has(interaction.source_observation_id) ||
        observationIds.has(interaction.settled_observation_id)
      ) {
        return scenarioAnchor(scenario) + "-step-" + (index + 1);
      }
    }
    return scenarioAnchor(scenario);
  }

  function cloneTemplate(templateId) {
    return document.getElementById(templateId).content.cloneNode(true);
  }

  function setText(elementId, value) {
    document.getElementById(elementId).textContent = value;
  }

  function findEvidence(runRecord, evidenceId) {
    return runRecord.evidence.find(function (evidence) {
      return evidence.artifact_id === evidenceId;
    });
  }

  function evidenceHref(path) {
    var pathSegments = String(path).split("/");
    return "../" + pathSegments.map(function (segment) {
      return encodeURIComponent(segment);
    }).join("/");
  }

  function journeyId(runRecord) {
    return runRecord.scenarios.length ? runRecord.scenarios[0].journey_id : runRecord.run_id;
  }

  function scenarioAnchor(scenario) {
    return "scenario-" + scenario.scenario_id;
  }

  function findingAnchor(scenario, findingId) {
    return scenarioAnchor(scenario) + "-finding-" + findingId;
  }

  function verdictTone(verdict) {
    if (verdict === "passed") {
      return "pass";
    }
    if (verdict === "failed") {
      return "fail";
    }
    return "warn";
  }

  function obligationTone(status) {
    if (status === "satisfied") {
      return "pass";
    }
    if (status === "unmet") {
      return "fail";
    }
    return "warn";
  }

  function percent(value, total) {
    return (value / total * 100).toFixed(4) + "%";
  }

  function formatDuration(seconds) {
    if (seconds < 60) {
      return seconds.toFixed(1) + " s";
    }
    return Math.floor(seconds / 60) + " min " + Math.floor(seconds % 60) + " s";
  }

  function formatTimestamp(value) {
    return new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
      timeZone: "UTC"
    }).format(new Date(value)) + " UTC";
  }

  function modelName(runRecord) {
    var names = Array.from(new Set(runRecord.model_calls.map(function (call) {
      return call.served_model || call.requested_model;
    }))).sort();
    if (!names.length) {
      return "Not recorded";
    }
    return names.length === 1
      ? names[0]
      : names[0] + " + " + (names.length - 1) + " more";
  }

  function deviceName(runRecord) {
    if (!runRecord.device) {
      return "Not recorded";
    }
    var platform = runRecord.device.platform;
    if (runRecord.device.operating_system_version) {
      platform += " " + runRecord.device.operating_system_version;
    }
    return (runRecord.device.model || runRecord.device.target_id) + " · " + platform;
  }

  function applicationName(runRecord) {
    if (!runRecord.application) {
      return "Not recorded";
    }
    var release = [runRecord.application.version, runRecord.application.build]
      .filter(Boolean)
      .join(" ");
    return release
      ? release + " · " + runRecord.application.application_id
      : runRecord.application.application_id;
  }

  function titleCase(value) {
    return String(value).replace(/_/g, " ").replace(/\b\w/g, function (letter) {
      return letter.toUpperCase();
    });
  }

  function applySavedTheme() {
    try {
      var savedTheme = localStorage.getItem("yewo-report-theme");
      if (savedTheme) {
        document.documentElement.setAttribute("data-theme", savedTheme);
      }
    } catch (error) {
      return;
    }
  }

  function connectThemeButton() {
    var button = document.querySelector("button.theme");
    if (!button) {
      return;
    }
    button.addEventListener("click", function () {
      var prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;
      var currentTheme = document.documentElement.getAttribute("data-theme") ||
        (prefersDark ? "dark" : "light");
      var nextTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      try {
        localStorage.setItem("yewo-report-theme", nextTheme);
      } catch (error) {
        return;
      }
    });
  }
})();
