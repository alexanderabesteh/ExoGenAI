import styles from './Header.module.css';

export function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <span className={styles.brandMark} aria-hidden="true" />
        <span className={styles.brandName}>ExoGen</span>
      </div>
      <nav className={styles.nav} aria-label="Primary">
        <a href="#explore" className={styles.link}>
          Explore
        </a>
        <a href="#data" className={styles.link}>
          Data
        </a>
        <a href="#export" className={styles.link}>
          Export
        </a>
      </nav>
    </header>
  );
}
