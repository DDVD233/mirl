#!/usr/bin/env bash
# Download the new medical-knowledge sources for the retrieval KB rebuild.
#
# Motivation (see RETRIEVAL_INVESTIGATION_2026-07-24.md): medical_knowledge_v2 is
# ~53% general Wikipedia + 42% PubMed abstract fragments, and 99% of its wikidoc
# rows are bare titles. HealthBench-Professional rewards current guideline text,
# exact drug dosing, consumer-health guidance and codes — none of which the KB
# holds. These four sources cover exactly those gaps:
#
#   statpearls   ~10k peer-reviewed clinical review articles (NCBI Bookshelf OA)
#   dailymed     FDA drug labels -> dosing / contraindications / warnings
#   medlineplus  NIH consumer-health topic summaries (patient-facing register)
#   icd10cm      the full 2026 ICD-10-CM code -> description table
#
# All four are public-domain / open-access US government or OA sources.
# Run once; re-running skips files that already exist at the expected size.
set -uo pipefail

OUT_DIR="${OUT_DIR:-/scratch/dvd/medkb_sources}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit 1

# fetch <url> <outfile> — skip if a non-empty file is already there.
fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "SKIP $out ($(du -h "$out" | cut -f1) already present)"
    return 0
  fi
  echo "GET  $out <- $url"
  if curl -sL --retry 3 --retry-delay 5 -o "$out.part" "$url"; then
    mv "$out.part" "$out"
    echo "DONE $out ($(du -h "$out" | cut -f1))"
  else
    echo "FAIL $out"
    rm -f "$out.part"
  fi
}

fetch "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip" icd10cm_2026.zip
fetch "https://medlineplus.gov/xml/mplus_topics_2026-07-25.xml" medlineplus_topics.xml
fetch "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz" statpearls.tar.gz

# DailyMed human prescription labels ship as 6 parts, ~3.2 GB each.
for i in 1 2 3 4 5 6; do
  fetch "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part${i}.zip" "dm_rx_part${i}.zip"
done

echo "ALL_DOWNLOADS_DONE"
