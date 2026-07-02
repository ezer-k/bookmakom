export default function Step3Confirm({ onNext }) {
  return (
    <section>
      <h2>Confirm book details</h2>
      <button type="button" onClick={onNext}>
        Save
      </button>
    </section>
  );
}
