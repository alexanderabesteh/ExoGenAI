import { Header } from '@/components/Header';
import styles from './page.module.css';

const planetSummaries = Array.from({ length: 12 }).map((_, index) => ({
  name: `Placeholder Planet ${String(index + 1).padStart(2, '0')}`,
  year: '20XX',
  method: 'Discovery Method',
  facility: 'Placeholder Observatory'
}));

export default function Home() {
  return (
    <>
      <Header />
      <main className={styles.main}>
        <section className={styles.hero}>
          <div className={styles.heroImageWrapper} aria-hidden="true" />
          <div className={styles.heroContent}>
            <span className={styles.heroSubtitle}>Current Catalogue</span>
            <h1 className={styles.heroTitle}>Exploring new worlds beyond our Solar System</h1>
            <p className={styles.heroDescription}>
              Browse a curated list of candidate planets, ready to be connected to the mission
              datasets you will integrate. The monochrome interface keeps attention on the data
              while leaving space for your future enhancements.
            </p>
          </div>
        </section>

        <section id="explore" className={styles.dashboard}>
          <div className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Explore the Planets</h2>
                <p className={styles.panelSubtitle}>
                  Scroll through the placeholder records and wire up your API when it is ready.
                </p>
              </div>
            </div>

            <div className={styles.searchBar}>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  d="M11 4a7 7 0 1 1-4.95 11.95l-2.5 2.5"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.7"
                />
              </svg>
              <input
                className={styles.searchInput}
                type="search"
                placeholder="Search exoplanet"
                aria-label="Search exoplanet"
                disabled
              />
            </div>

            <div className={styles.planetList}>
              {planetSummaries.map((planet) => (
                <article key={planet.name} className={styles.planetCard}>
                  <span className={styles.planetName}>{planet.name}</span>
                  <span className={styles.planetMeta}>
                    Discovery Method: {planet.method} · Discovery Year: {planet.year}
                  </span>
                  <span className={styles.planetMeta}>Discovery Facility: {planet.facility}</span>
                </article>
              ))}
            </div>
          </div>

          <div id="data" className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Exoplanet data (Visuals)</h2>
                <p className={styles.panelSubtitle}>
                  Future graph integrations will render mission telemetry in this canvas.
                </p>
              </div>
            </div>
            <div className={styles.dataPlaceholder}>
              <div>
                <strong>Reserved for interactive visuals</strong>
                <p>
                  Connect your analytics service to stream discovery metrics, orbital charts, or
                  comparative spectra directly into this panel.
                </p>
                <div className={styles.placeholderActions}>
                  <span className={styles.placeholderTag}>API Hook</span>
                  <span className={styles.placeholderTag}>Telemetry Feed</span>
                  <span className={styles.placeholderTag}>Render Surface</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="export" className={styles.exportBar}>
          <button type="button" className={styles.exportButton}>
            Export Data
          </button>
        </section>
      </main>
    </>
  );
}
