import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class GalleryService {
  constructor(private http: HttpClient) {}
  meta(card = ''): Observable<any> { return this.http.get('/api/meta', { params: { card } }); }
  // by='wall' keys on wall-clock t (playback); by='decision' keys on ingest step (the gallery as it was built)
  state(t: number, card = '', by = 'wall', step = -1): Observable<any> { return this.http.get('/api/state', { params: { t: Math.round(t), card, by, step } }); }
  detections(t: number, window = 400, card = '', by = 'wall', step = -1): Observable<any> { return this.http.get('/api/detections', { params: { t: Math.round(t), window, card, by, step } }); }
  decisions(frm: number, to: number, limit = 120, card = ''): Observable<any> { return this.http.get('/api/decisions', { params: { from: frm, to, limit, card } }); }
  next(t: number, dir: number, card = ''): Observable<any> { return this.http.get('/api/next', { params: { t, dir, card } }); }
  merges(): Observable<any> { return this.http.get('/api/merges'); }   // decision-order feed: consolidation events
  tracklet(seq: number): Observable<any> { return this.http.get('/api/tracklet/' + seq); }
  identity(gid: number): Observable<any> { return this.http.get('/api/identity/' + gid); }
  crop(detId: string): string { return '/api/crop/' + encodeURIComponent(detId); }
  embeddingProjection(mode = 'bank', card = '', t = 0, by = 'wall', step = -1): Observable<any> {
    // t must be an int — the backend (typed t: int) 422s on a float query value
    return this.http.get('/api/embedding_projection', { params: { mode, card, t: Math.round(t), by, step } });
  }
}
