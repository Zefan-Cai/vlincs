// plotly.js-dist-min ships no type declarations; this minimal ambient module lets the
// embedding-space scatter import it under strict TS. We only use react/purge + the plotly_click /
// plotly_hover event hooks, so a loose `any`-typed surface is sufficient.
declare module 'plotly.js-dist-min' {
  const Plotly: {
    react: (el: HTMLElement | string, data: any[], layout?: any, config?: any) => Promise<any>;
    newPlot: (el: HTMLElement | string, data: any[], layout?: any, config?: any) => Promise<any>;
    purge: (el: HTMLElement | string) => void;
    [k: string]: any;
  };
  export default Plotly;
}
