import { useState } from "react";
import Step1Photo from "./Step1_Photo.jsx";
import Step2Location from "./Step2_Location.jsx";
import Step4Success from "./Step4_Success.jsx";

const steps = [Step1Photo, Step2Location, Step4Success];

export default function UploadModal({ onClose }) {
  const [step, setStep] = useState(0);
  const Step = steps[step];

  return (
    <dialog open>
      <Step onNext={() => setStep((current) => Math.min(current + 1, steps.length - 1))} />
      <button type="button" onClick={onClose}>
        Close
      </button>
    </dialog>
  );
}
