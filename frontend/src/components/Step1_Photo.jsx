export default function Step1Photo({ onNext }) {
  return (
    <section>
      <h2>Upload photo</h2>
      <input type="file" accept="image/*" />
      <button type="button" onClick={onNext}>
        Continue
      </button>
    </section>
  );
}
