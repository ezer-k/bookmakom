export default function Step2Location({ onNext }) {
  return (
    <section>
      <h2>Set location</h2>
      <p>Drop a pin or use GPS fallback.</p>
      <button type="button" onClick={onNext}>
        Continue
      </button>
    </section>
  );
}
