import { Component } from '@angular/core';
import { GalleryComponent } from './gallery.component';
@Component({ selector: 'app-root', standalone: true, imports: [GalleryComponent], template: '<app-gallery/>' })
export class AppComponent {}
