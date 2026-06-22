import { Component, OnInit, AfterViewInit, ViewChildren, QueryList, ElementRef, HostListener, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NvKpiStripComponent } from 'novular';
import Plotly from 'plotly.js-dist-min';
import { GalleryService } from './gallery.service';

@Component({
  selector: 'app-gallery',
  standalone: true,
  imports: [CommonModule, FormsModule, NvKpiStripComponent],
  templateUrl: './gallery.component.html',
  styleUrl: './gallery.component.scss',
})
export class GalleryComponent implements OnInit, AfterViewInit {
  meta: any = null;
  cams: any[] = [];
  t = 0; t0 = 0; t1 = 1; scrubPos = 0; tq: number[] = [];   // tq = abs_ms quantiles (scrubber skips the Tc6/Tc8 gap)
  // timeline lens: 'wall' = wall-clock t (playback); 'decision' = ingest step (the gallery as it was BUILT,
  // one step per tracklet decision). sq = seq quantiles (the decision-order analogue of tq).
  mode: 'wall' | 'decision' = 'decision';   // default: replay the gallery in the order it was BUILT
  sq: number[] = []; step = 0; seq0 = 0; seq1 = 0; nDecisions = 0;
  playing = false; speed = 16; window = 400;
  dets: any[] = []; identities: any[] = []; decisions: any[] = []; allDecisions: any[] = []; allMerges: any[] = [];
  kpis: any[] = [];
  card = ''; cards: string[] = [];   // per-card toggle ('' = all cards; un-conflates Tc6/Tc8 in space & time)
  dataset = ''; datasetList: any[] = [];   // which gallery_<key> DB to view (the dataset switch); '' = backend default
  selGid: number | null = null; identityDetail: any = null;
  selSeq: number | null = null; trackletDetail: any = null; stripN = 12;   // how many tracklet crops to show
  // embedding-space panel: the 2D projection of the matcher's match space ("what the index sees")
  embMode: 'bank' | 'det' = 'bank';
  embPoints: any[] = []; embN = 0; embDim = 0; embEmpty = false;
  embHover: any = null;                 // hovered point -> identity info + crop thumbnail in the panel
  embExpanded = false;                  // the large zoomable dialog
  embLight = localStorage.getItem('embLight') === '1';   // light background for the embedding plot (persisted)
  private embDrawn = false; private embBigDrawn = false;
  private viewReady = false;
  private frameImgs: Record<string, HTMLImageElement> = {};  // last LOADED frame per VIDEO (double-buffer)
  private reqFrameIdx: Record<string, number> = {};          // frame index currently requested per VIDEO
  private scrubTimer: any = null;
  @ViewChildren('cam') canvases!: QueryList<ElementRef<HTMLCanvasElement>>;

  constructor(public api: GalleryService, private zone: NgZone) {}

  ngOnInit() {
    // pick the DB switch list first, then load. /datasets' `current` is the backend's default dataset.
    this.api.datasets().subscribe(
      (r) => { this.datasetList = r.datasets || []; this.dataset = r.current || r.default || ''; this.api.dataset = this.dataset; this.loadMeta(); },
      () => this.loadMeta(),                            // /datasets unavailable -> just load the default
    );
  }
  loadMeta() {
    this.api.meta(this.card).subscribe((m) => {
      this.meta = m; this.cams = m.cameras || []; this.t0 = m.t0; this.t1 = m.t1;
      this.tq = m.t_quantiles || []; this.cards = m.cards || [];
      this.sq = m.seq_quantiles || []; this.seq0 = m.seq0 || 0; this.seq1 = m.seq1 || 0; this.nDecisions = m.n_decisions || 0;
      this.scrubPos = 0; this.t = this.posToT(0); this.step = this.posToStep(0);
      // fetch the (card-scoped) decision history + the merge events (for the decision-order feed), then refresh
      this.api.merges().subscribe((mg) => {
        this.allMerges = mg || [];
        this.api.decisions(0, 1e15, 200000, this.card).subscribe((d) => {
          this.allDecisions = (d || []).slice().sort((a: any, b: any) => (a.abs_ms || 0) - (b.abs_ms || 0));
          this.refresh();
        });
      });
      this.loadEmbedding();
    });
  }

  // --- embedding-space panel: the matcher's match space (identity_reps.embedding_red) projected to 2D ---
  setEmbMode(m: 'bank' | 'det') { if (this.embMode !== m) { this.embMode = m; this.loadEmbedding(); } }
  // FULL reload — mode/card change: purge the old plot (drops its handlers) and re-fetch at current t.
  loadEmbedding() {
    this.embHover = null;
    const old = document.getElementById('embplot');
    if (old) Plotly.purge(old);   // drop the previous plot + its accumulated click/hover handlers
    this.embDrawn = false;
    this.refreshEmbedding();
  }
  // TIME re-fetch — called as the scrubber drives t: pull "the situation at t" (PCA coords are stable, the
  // returned subset just grows) and Plotly.react redraws in place WITHOUT purging the click/hover handlers.
  refreshEmbedding() {
    const obs = this.mode === 'decision'
      ? this.api.embeddingProjection(this.embMode, this.card, -1, 'decision', this.step)
      : this.api.embeddingProjection(this.embMode, this.card, this.t);
    obs.subscribe((r) => {
      this.embPoints = r.points || []; this.embN = r.n || 0; this.embDim = r.dim || 0;
      this.embEmpty = this.embN === 0;
      setTimeout(() => this.drawEmbedding(), 0);   // let *ngIf render #embplot before plotly grabs it
    });
  }
  private drawEmbedding() {
    this.drawEmbTo('embplot', false);
    if (this.embExpanded) this.drawEmbTo('embplotbig', true);   // keep the enlarged view in sync as you scrub
  }
  // Draw the embedding cloud into a target div. big=true is the dialog: modebar + scroll-zoom + axes, larger
  // markers; small is the inline panel. uirevision keeps zoom/pan across data refreshes (scrub doesn't reset it).
  private drawEmbTo(elId: string, big: boolean) {
    const el = document.getElementById(elId);
    const flag = big ? 'embBigDrawn' : 'embDrawn';
    if (!el || !this.embPoints.length) { if (el) Plotly.purge(el); (this as any)[flag] = false; return; }
    const lit = (g: number) => this.selGid === null || this.selGid === g;
    const sz = this.embMode === 'bank' ? (big ? 11 : 9) : (big ? 7 : 5);
    const trace: any = {
      x: this.embPoints.map((p) => p.x), y: this.embPoints.map((p) => p.y),
      mode: 'markers', type: 'scattergl',
      marker: { size: this.embPoints.map((p) => (this.selGid === p.gid ? sz + 5 : sz)),
                color: this.embPoints.map((p) => this.color(p.gid)),
                line: { width: this.embMode === 'bank' ? 0.6 : 0, color: 'rgba(0,0,0,0.5)' },
                opacity: this.embPoints.map((p) => (lit(p.gid) ? (this.embMode === 'bank' ? 0.95 : 0.6) : 0.12)) },
      customdata: this.embPoints.map((p) => p.gid),
      text: this.embPoints.map((p) => `id ${p.gid} · ${p.n_exemplars} ex · ${(p.cameras || []).join(',')}`),
      hovertemplate: '%{text}<extra></extra>',
    };
    const plotBg = this.embLight ? '#eef1f5' : '#0a0d10';
    const grid = this.embLight ? 'rgba(0,0,0,.10)' : 'rgba(120,140,170,.12)';
    const tick = this.embLight ? '#444' : '#9fb0c0';
    const layout: any = {
      margin: big ? { l: 28, r: 10, t: 8, b: 22 } : { l: 6, r: 6, t: 6, b: 6 },
      paper_bgcolor: 'transparent', plot_bgcolor: plotBg, showlegend: false, uirevision: 'emb',
      xaxis: { showgrid: big, gridcolor: grid, zeroline: false, showticklabels: big, tickfont: { color: tick } },
      yaxis: { showgrid: big, gridcolor: grid, zeroline: false, showticklabels: big, tickfont: { color: tick } },
      ...(big ? { autosize: true } : { height: 300 }),
    };
    const config: any = big
      ? { displayModeBar: true, scrollZoom: true, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }
      : { displayModeBar: false, responsive: true };
    Plotly.react(el, [trace], layout, config).then(() => {
      if ((this as any)[flag]) return;
      (this as any)[flag] = true;
      (el as any).on('plotly_click', (e: any) => {
        const g = e.points?.[0]?.customdata;
        if (g != null) this.zone.run(() => this.selectGid(+g));   // re-enter Angular's zone (plotly fires outside it)
      });
      (el as any).on('plotly_hover', (e: any) => {
        const i = e.points?.[0]?.pointIndex;
        if (i != null) this.zone.run(() => this.embHover = this.embPoints[i]);
      });
    });
  }
  toggleEmbLight() { this.embLight = !this.embLight; localStorage.setItem('embLight', this.embLight ? '1' : '0'); this.drawEmbedding(); }
  openEmbDialog() { this.embExpanded = true; this.embBigDrawn = false; setTimeout(() => this.drawEmbedding(), 0); }
  closeEmbDialog() { const el = document.getElementById('embplotbig'); if (el) Plotly.purge(el); this.embBigDrawn = false; this.embExpanded = false; }
  selectCard(c: string) {
    if (this.card === c) return;
    this.card = c; this.playing = false;
    this.selGid = null; this.identityDetail = null; this.selSeq = null; this.trackletDetail = null;
    this.frameImgs = {}; this.reqFrameIdx = {};   // canvases change with the card -> reset the double-buffer
    this.loadMeta();
  }
  // Switch which gallery_<key> DB we view. Resets card/selection (cards + ids differ per dataset) and reloads.
  selectDataset(d: string) {
    if (!d || this.dataset === d) return;
    this.dataset = d; this.api.dataset = d;
    this.card = ''; this.playing = false;
    this.selGid = null; this.identityDetail = null; this.selSeq = null; this.trackletDetail = null;
    this.frameImgs = {}; this.reqFrameIdx = {};   // canvases change with the dataset -> reset the double-buffer
    this.loadMeta();
  }
  trackSeq(_i: number, d: any) { return d.isMerge ? 'm' + d.merge_id : 'd' + d.seq; }   // merges + decisions can share a seq
  selectMerge(d: any) { this.selSeq = null; this.trackletDetail = null; this.selectGid(d.new_gid); }   // inspect the survivor
  ngAfterViewInit() { this.viewReady = true; }

  refresh() {
    if (this.mode === 'decision') { this.refreshDecision(); return; }
    const t = Math.round(this.t);   // backend endpoints type t as int — a float query value 422s
    this.t = t;
    this.api.state(t, this.card).subscribe((s) => {
      this.identities = s.identities || [];
      const xc = this.identities.filter((i: any) => (i.cameras || []).length > 1).length;
      const seen = this.identities.reduce((a: number, i: any) => a + (+i.seen || 0), 0);
      this.kpis = [
        { label: 'identities so far', value: s.n_identities, emphasis: true },
        { label: 'cross-camera', value: xc },
        { label: 'dets assigned', value: seen },
        { label: 'time', value: ((t - this.t0) / 1000).toFixed(1) + 's / ' + ((this.t1 - this.t0) / 1000).toFixed(0) + 's' },
      ];
    });
    this.api.detections(t, this.window, this.card).subscribe((d) => { this.dets = d.dets || []; this.render(); });
    // all decisions up to t, newest first (index 0 = the decision being made "now")
    this.decisions = this.allDecisions.filter((x) => (x.abs_ms || 0) <= t).reverse();
    this.refreshEmbedding();   // time-sync the embedding panel to t (grows as the scrubber advances)
    if (this.selGid != null) this.loadIdentity();   // keep the identity bank cursor-consistent
  }

  // Decision-order lens: the gallery AS OF ingest step N. Identities/bank/dets all gated on seq<=step, so
  // stepping replays how the gallery was actually built (one tracklet decision per step), and the canvases
  // show the tracklet decided at this step (not a wall-clock frame). No "future crops".
  refreshDecision() {
    const step = this.step = Math.round(this.posToStep(this.scrubPos));
    const cur = this.allDecisions.find((x) => x.seq === step);
    if (cur && cur.abs_ms != null) this.t = cur.abs_ms;   // frame context for the OTHER cameras' canvases
    this.api.state(-1, this.card, 'decision', step).subscribe((s) => {
      this.identities = s.identities || [];
      const xc = this.identities.filter((i: any) => (i.cameras || []).length > 1).length;
      const seen = this.identities.reduce((a: number, i: any) => a + (+i.seen || 0), 0);
      this.kpis = [
        { label: 'identities so far', value: s.n_identities, emphasis: true },
        { label: 'cross-camera', value: xc },
        { label: 'dets committed', value: seen },
        { label: 'decision', value: 'step ' + step + ' / ' + Math.max(0, this.nDecisions - 1) },
      ];
    });
    this.api.detections(-1, this.window, this.card, 'decision', step).subscribe((d) => { this.dets = d.dets || []; this.render(); });
    // feed: decisions + merge events up to step N, newest first. A merge at at_seq=S sorts just AFTER the
    // decision at seq=S (sortval S+0.5), so it reads as "...then the resolve merged these".
    const decs = this.allDecisions.filter((x) => (x.seq ?? -1) <= step).map((x) => ({ ...x, isMerge: false, sortval: x.seq }));
    const mrgs = this.allMerges.filter((m) => m.at_seq <= step).map((m) => ({
      isMerge: true, seq: m.at_seq, sortval: m.at_seq + 0.5, decision_type: 'merged',
      old_gid: m.old_gid, new_gid: m.new_gid, merge_id: m.merge_id, score: m.score }));
    this.decisions = [...decs, ...mrgs].sort((a, b) => (b.sortval - a.sortval) || ((b.merge_id || 0) - (a.merge_id || 0)));
    this.refreshEmbedding();
    if (this.selGid != null) this.loadIdentity();   // keep the identity bank consistent with the step
    // auto-open the current decision's tracklet (step IS a seq) — guarded so a drag doesn't refetch each tick
    if (this.selSeq !== step && this.allDecisions.some((x) => x.seq === step)) this.selectTracklet(step);
  }

  render() {
    if (!this.viewReady || !this.canvases) return;
    // Index-independent: each canvas carries its OWN video (data-video) and we look up its cam by that.
    // So a card switch (cams 10->6) can never paint a canvas with another video's frame/boxes even if the
    // ViewChildren list and the cams array momentarily disagree on order.
    this.canvases.forEach((ref) => {
      const cv = ref.nativeElement;
      const v = cv.dataset['video'];
      if (!v) return;
      const cam = this.cams.find((c) => c.video === v);
      if (!cam) return;
      const camDets = this.dets.filter((d: any) => d.video === v);
      const fidx = camDets.length ? Math.max(...camDets.map((d: any) => d.frame_idx))
                                  : Math.max(0, Math.round((this.t - (cam.start_ms || 0)) / 1000 * 30));
      // request the new frame only when it changed; KEEP showing the last loaded frame until it's ready
      if (this.reqFrameIdx[v] !== fidx) {
        this.reqFrameIdx[v] = fidx;
        const img = new Image();
        img.onload = () => { this.frameImgs[v] = img; this.paintCam(cv, cam); };  // swap in only when loaded
        img.src = `/api/frame/${encodeURIComponent(v)}?frame=${fidx}&w=${cv.width}${this.api.dataset ? '&dataset=' + encodeURIComponent(this.api.dataset) : ''}`;
      }
      this.paintCam(cv, cam);   // draw immediately with the last loaded frame (no black flash) + current boxes
    });
  }

  private paintCam(cv: HTMLCanvasElement, cam: any) {
    const cx = cv.getContext('2d'); if (!cx) return;
    const img = this.frameImgs[cam.video];
    cx.globalAlpha = 1;
    if (img && img.complete && img.naturalWidth) cx.drawImage(img, 0, 0, cv.width, cv.height);
    else { cx.fillStyle = '#0a0d10'; cx.fillRect(0, 0, cv.width, cv.height); }
    const sx = cv.width / (cam.w || 1920), sy = cv.height / (cam.h || 1080);
    for (const d of this.dets) {
      if (d.video !== cam.video) continue;
      const sel = this.selGid === null || this.selGid === d.gid;
      cx.globalAlpha = sel ? 1 : 0.18; cx.strokeStyle = this.color(d.gid); cx.lineWidth = this.selGid === d.gid ? 2.5 : 1.4;
      cx.strokeRect(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sy);
      if (sel && (d.x2 - d.x1) * sx > 24) { cx.fillStyle = this.color(d.gid); cx.font = 'bold 11px sans-serif'; cx.fillText(String(d.gid), d.x1 * sx + 2, d.y1 * sy - 2); }
    }
    cx.globalAlpha = 1;
  }

  // scrubber maps over detection-time QUANTILES so it skips the ~2.5h Tc6/Tc8 dead gap (74% of linear t)
  posToT(pos: number): number {
    const q = this.tq;
    if (!q || q.length < 2) return Math.round(this.t0 + (this.t1 - this.t0) * pos / 1000);
    const f = Math.max(0, Math.min(1, pos / 1000)) * (q.length - 1), i = Math.floor(f);
    return Math.round(q[i] + (q[Math.min(i + 1, q.length - 1)] - q[i]) * (f - i));
  }
  tToPos(t: number): number {
    const q = this.tq;
    if (!q || q.length < 2) return 1000 * (t - this.t0) / ((this.t1 - this.t0) || 1);
    if (t <= q[0]) return 0;
    if (t >= q[q.length - 1]) return 1000;
    let i = 0; while (i < q.length - 1 && q[i + 1] < t) i++;
    return 1000 * (i + (t - q[i]) / ((q[i + 1] - q[i]) || 1)) / (q.length - 1);
  }
  // seq (decision-order) analogues of posToT/tToPos — map the scrubber over ingest step
  posToStep(pos: number): number {
    const q = this.sq;
    if (!q || q.length < 2) return Math.round(this.seq0 + (this.seq1 - this.seq0) * pos / 1000);
    const f = Math.max(0, Math.min(1, pos / 1000)) * (q.length - 1), i = Math.floor(f);
    return Math.round(q[i] + (q[Math.min(i + 1, q.length - 1)] - q[i]) * (f - i));
  }
  stepToPos(step: number): number {
    const q = this.sq;
    if (!q || q.length < 2) return 1000 * (step - this.seq0) / ((this.seq1 - this.seq0) || 1);
    if (step <= q[0]) return 0;
    if (step >= q[q.length - 1]) return 1000;
    let i = 0; while (i < q.length - 1 && q[i + 1] < step) i++;
    return 1000 * (i + (step - q[i]) / ((q[i + 1] - q[i]) || 1)) / (q.length - 1);
  }
  setMode(m: 'wall' | 'decision') {
    if (this.mode === m) return;
    this.mode = m; this.playing = false; this.scrubPos = 0;
    this.t = this.posToT(0); this.step = this.posToStep(0);
    this.loadEmbedding();   // the panel re-keys (by=wall|decision) -> full reload, then refresh the rest
    this.refresh();
  }
  onScrub() {
    if (this.mode === 'decision') this.step = this.posToStep(this.scrubPos);
    else this.t = this.posToT(this.scrubPos);
    if (this.scrubTimer) clearTimeout(this.scrubTimer);
    this.scrubTimer = setTimeout(() => this.refresh(), 60);   // debounce: one fetch when the drag settles
  }
  stepDecision(dir: number) {
    this.playing = false;
    if (this.mode === 'decision') {                 // step to the next/prev actual decision seq
      const seqs = this.allDecisions.map((x) => x.seq).sort((a: number, b: number) => a - b);
      const nxt = dir >= 0 ? seqs.find((s: number) => s > this.step)
                           : [...seqs].reverse().find((s: number) => s < this.step);
      if (nxt != null) { this.step = nxt; this.scrubPos = this.stepToPos(nxt); this.refresh(); }
      return;
    }
    this.api.next(this.t, dir, this.card).subscribe((r) => {
      if (r && r.abs_ms != null) {
        this.t = r.abs_ms;
        this.scrubPos = this.tToPos(this.t);
        this.refresh();
        this.selectTracklet(r.seq);          // open its what/why inspector
      }
    });
  }
  togglePlay() { this.playing = !this.playing; if (this.playing) this.tick(); }
  tick() {
    if (!this.playing) return;
    this.scrubPos = Math.min(1000, this.scrubPos + Math.max(1, this.speed / 8));   // advance in quantile space
    if (this.mode === 'decision') this.step = this.posToStep(this.scrubPos);
    else this.t = this.posToT(this.scrubPos);
    if (this.scrubPos >= 1000) this.playing = false;
    this.refresh();
    if (this.playing) setTimeout(() => this.tick(), 130);
  }

  selGidRelated: Set<number> | null = null;   // selGid + every gid that merged into it (for feed highlighting)
  selectGid(g: number) {
    this.selGid = this.selGid === g ? null : g; this.identityDetail = null; this.selSeq = null; this.trackletDetail = null;
    this.selGidRelated = this.selGid === null ? null : this.relatedGids(this.selGid);
    this.loadIdentity();
    this.render();
    this.drawEmbedding();   // re-tint the cloud so the focused id pops
  }
  // fetch the selected identity AS OF the current cursor, so its bank/counts agree with the embedding + KPIs
  private loadIdentity() {
    if (this.selGid == null) { this.identityDetail = null; return; }
    const obs = this.mode === 'decision'
      ? this.api.identity(this.selGid, 'decision', this.step)
      : this.api.identity(this.selGid, 'wall', -1, this.t);
    obs.subscribe((d) => this.identityDetail = d);
  }
  // gid + all gids that canonicalize to it via merges (a survivor "is" its merged-away ids too)
  private relatedGids(gid: number): Set<number> {
    const parent: Record<number, number> = {};
    for (const m of this.allMerges) parent[m.old_gid] = m.new_gid;
    const canon = (x: number) => { const seen = new Set<number>(); while (parent[x] != null && !seen.has(x)) { seen.add(x); x = parent[x]; } return x; };
    const set = new Set<number>([gid]);
    for (const k of Object.keys(parent)) { const o = +k; if (canon(o) === gid) set.add(o); }
    return set;
  }
  // is a feed row "involved" with the selected identity? — chose it, considered it, or (merge) touches it
  feedInvolved(d: any): boolean {
    const rel = this.selGidRelated; if (!rel) return false;
    if (d.isMerge) return rel.has(d.old_gid) || rel.has(d.new_gid);
    if (rel.has(d.chosen_gid)) return true;
    return (d.candidate_gids || []).some((g: number) => rel.has(g));
  }
  selectTracklet(seq: number) { this.selSeq = seq; this.stripN = 12; this.api.tracklet(seq).subscribe((d) => this.trackletDetail = d); }
  // evenly-spaced subsample of the tracklet's crops at the current stripN (the endpoint already returns ALL
  // dets; we just choose how many to render). "load more" grows stripN; "show all" jumps to every det.
  trackletStrip(): any[] {
    const dets = this.trackletDetail?.dets || [];
    const n = Math.min(this.stripN, dets.length);
    if (n <= 0) return [];
    if (n >= dets.length) return dets;
    const idx = new Set<number>();
    for (let k = 0; k < n; k++) idx.add(Math.round(k * (dets.length - 1) / (n - 1)));
    return [...idx].sort((a, b) => a - b).map((i) => dets[i]);
  }
  loadMoreCrops(all = false) {
    const total = this.trackletDetail?.n_dets || 0;
    this.stripN = all ? total : Math.min(total, this.stripN + 36);
  }
  // --- camera geo mini-map (positions from meta.cameras lat/lon; edges = empirically-overlapping FOV pairs) ---
  overlapPairs: [string, string][] = [
    ['MCAM00','MCAM05'],['MCAM00','MCAM06'],['MCAM00','MCAM08'],['MCAM03','MCAM06'],
    ['MCAM03','MCAM08'],['MCAM04','MCAM08'],['MCAM05','MCAM06'],['MCAM05','MCAM08'],['MCAM06','MCAM08'],
  ];
  private geoB(): any {
    const la = this.cams.map((c) => +c.lat).filter((v) => isFinite(v));
    const lo = this.cams.map((c) => +c.lon).filter((v) => isFinite(v));
    return { la0: Math.min(...la), la1: Math.max(...la), lo0: Math.min(...lo), lo1: Math.max(...lo) };
  }
  // NB: reject null/undefined lat/lon explicitly — `+null === 0` is finite, so without this a geo-less
  // dataset (e.g. MS02, which ships no camera extrinsics) would falsely pass and stack every camera at
  // (0,0), rendering as a single point. With the guard the map correctly hides instead.
  hasGeo() {
    return this.cams.length > 0 && this.cams.every(
      (c) => c.lat != null && c.lon != null && isFinite(+c.lat) && isFinite(+c.lon));
  }
  readonly MAP_W = 230; readonly MAP_H = 160;
  camPt(c: any): { x: number; y: number } {
    const b = this.geoB(), pad = 22;
    const x = pad + (this.MAP_W - 2 * pad) * ((+c.lon - b.lo0) / ((b.lo1 - b.lo0) || 1));
    const y = pad + (this.MAP_H - 2 * pad) * (1 - (+c.lat - b.la0) / ((b.la1 - b.la0) || 1));
    return { x, y };
  }
  camByName(n: string) { return this.cams.find((c) => c.camera === n); }
  mapEdges(): { x1: number; y1: number; x2: number; y2: number }[] {
    const out: any[] = [];
    for (const [a, b] of this.overlapPairs) {
      const ca = this.camByName(a), cb = this.camByName(b);
      if (ca && cb) { const pa = this.camPt(ca), pb = this.camPt(cb); out.push({ x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y }); }
    }
    return out;
  }
  private selCams(): string[] { return (this.identityDetail?.cameras || []) as string[]; }
  camSpansSel(c: any) { return this.selGid !== null && this.selCams().includes(c.camera); }

  @HostListener('document:keydown', ['$event'])
  onKey(e: KeyboardEvent) {
    if (e.key === 'Escape' && this.embExpanded) { this.closeEmbDialog(); e.preventDefault(); return; }
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight') { this.stepDecision(1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { this.stepDecision(-1); e.preventDefault(); }
    else if (e.key === ' ') { this.togglePlay(); e.preventDefault(); }
    else if (e.key === ']') { const o = [200, 400, 1000]; this.window = o[Math.min(o.length - 1, o.indexOf(this.window) + 1)] || this.window; this.refresh(); }
    else if (e.key === '[') { const o = [200, 400, 1000]; this.window = o[Math.max(0, o.indexOf(this.window) - 1)] || this.window; this.refresh(); }
  }

  color(g: number) { return `hsl(${(g * 47) % 360} 75% 55%)`; }
  crop(d: string) { return this.api.crop(d); }
  shortCard(v: string) { const p = (v || '').split('_').pop() || ''; return p.split('-').pop() || p; }   // ...2024-03-Tc6 -> Tc6
  // distinct camera NAMES (the geo map is per physical camera; canvases are per video)
  distinctCams() { const seen = new Set<string>(); return this.cams.filter((c) => !seen.has(c.camera) && seen.add(c.camera)); }
  // crop loading state: shimmer until the JPEG paints (.ld) or 404s (.er); trackBy keeps the shimmer
  // honest when switching identities (new det -> new <img> element, not a reused stale-class one).
  onImg(e: Event, ok: boolean) { const el = e.target as HTMLElement; el.classList.remove('ld', 'er'); el.classList.add(ok ? 'ld' : 'er'); }
  trackImg = (_: number, x: any) => x?.rep_det ?? x?.det_id ?? _;
  trackVid = (_: number, c: any) => c?.video ?? _;   // stable canvas identity per video across card switches
  secs(ms: number) { return ((ms - this.t0) / 1000).toFixed(1); }
  detLabel(detId: string) { return (detId || '').split('::').pop() || detId; }   // ...::MCAM310:245:1 -> MCAM310:245:1
  filterVeto(arr: string[]) { return [...new Set((arr || []).filter((v) => v && v !== 'below_tau'))]; }
  // the decision config the CURRENT DB state was built with (persisted as a role='gallery' models row)
  galleryConfig(): any { return (this.meta?.models || []).find((m: any) => m.role === 'gallery')?.params || null; }
  // clean display labels (+ ordering) for the raw config keys
  private static CFG: { k: string; label: string }[] = [
    { k: 'cannot_link', label: 'Vetoes' }, { k: 'match_mode', label: 'Match mode' },
    { k: 'tau', label: 'Match τ' }, { k: 'merge_tau', label: 'Merge τ' }, { k: 'admit_tau', label: 'Admit τ' },
    { k: 'coherence_floor', label: 'Coherence floor' }, { k: 'tracklet_coh_min', label: 'Min tracklet coherence' },
    { k: 'max_reps', label: 'Bank cap' }, { k: 'max_speed', label: 'Max speed' },
    { k: 'sim_window_ms', label: 'Sim window' }, { k: 'same_box_iou', label: 'Same-box IoU' },
  ];
  private fmtCfg(k: string, v: any): string {
    if (k === 'cannot_link') return v ? 'on' : 'off';
    if (k === 'sim_window_ms') return `${v} ms`;
    if (k === 'max_speed') return `${v} m/s`;
    return String(v);
  }
  // one distinctive tooltip per knob — what it controls, not just its name
  private static CFG_TIPS: Record<string, string> = {
    cannot_link: 'Physical vetoes — block matches that can’t happen: same_frame (two distinct boxes in one frame), simultaneity & travel (one person in two far-apart cameras at once). On = enforced; off = appearance-only.',
    match_mode: 'How a candidate id is scored against a new tracklet — centroid: cosine to the id’s whole-bank mean; max: nearest single exemplar; retrieval: FAISS k-NN vote.',
    tau: 'Match threshold — cosine ≥ τ to an existing id ⇒ match; otherwise expand into a new id.',
    merge_tau: 'Resolve threshold — during resolve(), two ids whose exemplar centroids agree at ≥ this cosine are consolidated into one.',
    admit_tau: 'Bank redundancy cutoff — a matched tracklet becomes a new exemplar only if it is at most this similar to existing ones (too-similar = redundant, skipped).',
    coherence_floor: 'Anti-accretion cutoff — drop a would-be exemplar sitting farther than this from its bank; stops one id becoming a “matches-everything” attractor.',
    tracklet_coh_min: 'Quarantine cutoff — a tracklet whose own internal coherence is below this gets a solo id and never joins a matchable bank (likely an ID-switch / mixed people).',
    max_reps: 'Bank cap — the most exemplars kept per identity in the match bank.',
    max_speed: 'Travel veto — a cross-camera match implying ground speed above this (m/s) between the two cameras is blocked as impossible.',
    sim_window_ms: 'Simultaneity slop — detections in two cameras within this many ms are treated as the same instant for the simultaneity veto.',
    same_box_iou: 'Same-frame cutoff — two boxes in one frame below this IoU are judged different people (can’t share an id); above it = the same person the tracker duplicated.',
  };
  cfgTip(k: string): string { return GalleryComponent.CFG_TIPS[k] || k; }
  configEntries(): { k: string; label: string; v: string }[] {
    const c = this.galleryConfig(); if (!c) return [];
    const known = GalleryComponent.CFG.filter((e) => e.k in c);
    const extra = Object.keys(c).filter((k) => !GalleryComponent.CFG.some((e) => e.k === k)).map((k) => ({ k, label: k }));
    return [...known, ...extra].map((e) => ({ k: e.k, label: e.label, v: this.fmtCfg(e.k, c[e.k]) }));
  }
  addedTracklets() { return (this.identityDetail?.tracklets || []).filter((t: any) => t.admitted); }
  rejectedTracklets() { return (this.identityDetail?.tracklets || []).filter((t: any) => !t.admitted); }

  // The vetoed candidates for the open tracklet, each with the on-demand supporting numbers the backend
  // computed (n_shared, median IoU, attractor profile / travel dist+dt+speed). One self-explanatory line each.
  vetoLines(): any[] { return this.trackletDetail?.veto_explain || []; }
  fmtVeto(v: any): string {
    const s = (v.score != null) ? (+v.score).toFixed(3) : '?';
    if (v.kind === 'same_frame') {
      const med = (v.median_box_iou != null) ? (+v.median_box_iou).toFixed(2) : '?';
      return `Blocked from id ${v.gid} (cosine ${s}): shares ${v.n_shared}/${v.n_track} of this track's frames`
        + ` at median box-IoU ${med} → spatially distinct. id ${v.gid} = ${v.gid_n_tracklets} tracklets`
        + ` across frames ${v.gid_frame_min}–${v.gid_frame_max} / ${v.gid_n_cameras} cam(s)`
        + ` — likely over-merged.`;
    }
    if (v.kind === 'travel') {
      const dist = (v.dist_m != null) ? (+v.dist_m).toFixed(0) : '?';
      const dt = (v.dt_s != null) ? (+v.dt_s).toFixed(1) : '?';
      const sp = (v.speed_ms != null) ? (+v.speed_ms).toFixed(1) : '?';
      return `Blocked from id ${v.gid} (travel): id ${v.gid} last on ${v.from_camera} @f${v.from_frame};`
        + ` this track on ${v.to_camera} @f${v.to_frame} — ${dist} m in ${dt} s = ${sp} m/s`
        + ` > ${v.max_speed} m/s.`;
    }
    // simultaneity (or any other physical veto) — fall back to the raw reason + cosine
    return `Blocked from id ${v.gid} (${v.veto_reason}) at cosine ${s}.`;
  }
  whyText(dc: any): string {
    if (!dc) return '';
    if (dc.decision_type === 'merged') {
      return `Merged id ${dc.candidate_gids?.[0]} → id ${dc.chosen_gid} during deferred consolidation: their exemplar centroids were cosine ${dc.scores?.[0]} ≥ merge_tau ${dc.threshold} (and cannot-link allowed it). Those dets now share id ${dc.chosen_gid}.`;
    }
    if (dc.decision_type === 'quarantine') {
      if (dc.candidate_gids?.length) {   // ambiguous-expand quarantine: resembled an existing id but was blocked
        const b = (dc.veto_reasons || []).find((r: string) => r && r !== 'below_tau') || 'cannot-link';
        return `Quarantined instead of expanding: best match was id ${dc.candidate_gids[0]} at sim ${dc.scores?.[0]}, but it was blocked (${b}). Spawning a NEW identity here would be premature — it clearly resembles an existing one — so it's held as its own id ${dc.chosen_gid}, NOT added to any bank, for consolidation to reunite later.`;
      }
      return `Quarantined ("do nothing"): the tracklet's internal self-coherence is ${dc.scores?.[0]} — too inconsistent (likely an ID-switch / mixed people), so it gets its own id ${dc.chosen_gid} and is NOT added to any matchable bank. It can neither seed nor pollute an identity.`;
    }
    const gids = dc.candidate_gids || [], sims = dc.scores || [], reasons = dc.veto_reasons || [];
    if (dc.decision_type === 'match') {
      const blocked = gids.map((g: number, i: number) => (reasons[i] && reasons[i] !== 'below_tau') ? `id ${g}→${reasons[i]}` : null).filter(Boolean);
      return `Matched existing id ${dc.chosen_gid} at sim ${dc.scores?.[0]} ≥ τ=${dc.threshold}.`
        + (blocked.length ? ` (first ruled out: ${blocked.join(', ')})` : '')
        + (dc.admitted ? ' Added to the exemplar bank.' : ' ' + this.notAddedText(dc));
    }
    // expand: explain, per candidate, why none was taken
    const parts: string[] = [];
    for (let i = 0; i < gids.length; i++) {
      const r = reasons[i];
      if (r === 'below_tau') parts.push(`id ${gids[i]} (sim ${sims[i]}) < τ=${dc.threshold}`);
      else if (r) parts.push(`id ${gids[i]} (sim ${sims[i]}) blocked: ${r}`);
    }
    const why = gids.length ? parts.join('; ') : 'no existing identity to compare against';
    return `Spawned a NEW identity (id ${dc.chosen_gid}) — ${why}. First exemplar added to the bank.`;
  }

  // Tooltip for an ADDED exemplar crop in the identity bank panel.
  exTitle(tk: any): string {
    return `Bank exemplar — ${tk.camera}, ${tk.decision_type} at ${this.secs(tk.abs_ms)}s, ${tk.n_dets} dets. Click to inspect this tracklet.`;
  }
  // Compact per-tracklet tooltip for a rejected (not-added) exemplar in the identity panel:
  // base "<cam> <type> · <s>s · <n> dets" plus the specific gate + deciding number.
  rejTitle(tk: any): string {
    const base = tk.camera + ' ' + tk.decision_type + ' · ' + this.secs(tk.abs_ms) + 's · ' + tk.n_dets + ' dets';
    switch (tk?.admit_reason) {
      case 'redundant':
        return base + ` · REDUNDANT: closest exemplar cosine ${tk.admit_sim} ≥ admit_tau ${tk.admit_tau}`;
      case 'incoherent':
        return base + ` · TOO FAR: nearest-bank cosine ${tk.admit_min} < coherence_floor ${tk.coherence_floor}`;
      case 'bank_full':
        return base + ` · BANK FULL: id already has ${tk.max_reps} exemplars`;
      default:
        return base + ' · not added (too similar to an existing exemplar)';
    }
  }

  // WHY a matched tracklet's pooled vector was NOT admitted to the exemplar bank — the specific
  // diversity gate that fired + its deciding number. Uses the additive decision_log fields.
  notAddedText(dc: any): string {
    switch (dc?.admit_reason) {
      case 'redundant':
        return `Not added (REDUNDANT): closest exemplar cosine ${dc.admit_sim} ≥ admit_tau ${dc.admit_tau}`
          + ` — a near-duplicate, adds no new view.`;
      case 'incoherent':
        return `Not added (TOO FAR): nearest-bank cosine ${dc.admit_min} < coherence_floor ${dc.coherence_floor}`
          + ` — too dissimilar, kept out to avoid an attractor.`;
      case 'bank_full':
        return `Not added (BANK FULL): id already has ${dc.max_reps} exemplars.`;
      default:
        return 'Not added — too similar to an existing exemplar.';
    }
  }
}
