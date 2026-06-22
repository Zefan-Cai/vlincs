import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class GalleryService {
  // Active dataset key ('' -> backend default). Set by the dataset switch and sent on EVERY request (incl.
  // the <img src> crop/frame URLs) so one viz backend serves any gallery_<key> DB without a restart.
  dataset = '';
  constructor(private http: HttpClient) {}
  private p(extra: any = {}): any { return this.dataset ? { ...extra, dataset: this.dataset } : extra; }
  private dq(): string { return this.dataset ? '?dataset=' + encodeURIComponent(this.dataset) : ''; }

  datasets(): Observable<any> { return this.http.get('/api/datasets'); }   // the switch list (no dataset param)
  meta(card = ''): Observable<any> { return this.http.get('/api/meta', { params: this.p({ card }) }); }
  // by='wall' keys on wall-clock t (playback); by='decision' keys on ingest step (the gallery as it was built)
  state(t: number, card = '', by = 'wall', step = -1): Observable<any> { return this.http.get('/api/state', { params: this.p({ t: Math.round(t), card, by, step }) }); }
  detections(t: number, window = 400, card = '', by = 'wall', step = -1): Observable<any> { return this.http.get('/api/detections', { params: this.p({ t: Math.round(t), window, card, by, step }) }); }
  decisions(frm: number, to: number, limit = 120, card = ''): Observable<any> { return this.http.get('/api/decisions', { params: this.p({ from: frm, to, limit, card }) }); }
  next(t: number, dir: number, card = ''): Observable<any> { return this.http.get('/api/next', { params: this.p({ t, dir, card }) }); }
  merges(): Observable<any> { return this.http.get('/api/merges', { params: this.p() }); }   // decision-order feed: consolidation events
  tracklet(seq: number): Observable<any> { return this.http.get('/api/tracklet/' + seq, { params: this.p() }); }
  // per-vetoed-candidate supporting numbers for a decision (also embedded on /tracklet/{seq})
  vetoExplain(seq: number): Observable<any> { return this.http.get('/api/veto_explain/' + seq, { params: this.p() }); }
  identity(gid: number, by = 'wall', step = -1, t = -1): Observable<any> {
    return this.http.get('/api/identity/' + gid, { params: this.p({ by, step, t: Math.round(t) }) });
  }
  crop(detId: string): string { return '/api/crop/' + encodeURIComponent(detId) + this.dq(); }
  embeddingProjection(mode = 'bank', card = '', t = 0, by = 'wall', step = -1): Observable<any> {
    // t must be an int — the backend (typed t: int) 422s on a float query value
    return this.http.get('/api/embedding_projection', { params: this.p({ mode, card, t: Math.round(t), by, step }) });
  }
}
