#include <vector>
#include "CoarseBiLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"

using namespace std;

namespace Moses {
int CoarseBiLMState::Compare(const FFState& other) const {
	const CoarseBiLMState &otherState =
			static_cast<const CoarseBiLMState&>(other);

	if (m_targetLen == otherState.m_targetLen)
		return 0;
	return (m_targetLen < otherState.m_targetLen) ? -1 : +1;
}

////////////////////////////////////////////////////////////////
CoarseBiLM::CoarseBiLM(const std::string &line) :
		StatefulFeatureFunction(3, line) {
	ReadParameters();
}

FFState* CoarseBiLM::EvaluateWhenApplied(const Hypothesis& cur_hypo,
		const FFState* prev_state,
		ScoreComponentCollection* accumulator) const {
	// dense scores
	vector<float> newScores(m_numScoreComponents);
	newScores[0] = 1.5;
	newScores[1] = 0.3;
	newScores[2] = 0.4;
	accumulator->PlusEquals(this, newScores);

	// sparse scores
	accumulator->PlusEquals(this, "sparse-name", 2.4);

	TargetPhrase currTargetPhrase = cur_hypo.GetCurrTargetPhrase();
	AlignmentInfo alignTerm = currTargetPhrase.GetAlignTerm();
	AlignmentInfo::CollType alignments = alignTerm.GetAlignments();
	string sourcePhraseStringRep = cur_hypo.GetSourcePhraseStringRep();

	// int targetLen = cur_hypo.GetCurrTargetPhrase().GetSize(); // ??? [UG]
	return new CoarseBiLMState(0);
}

void CoarseBiLM::SetParameter(const std::string& key,
		const std::string& value) {
	if (key == "arg") {
		// set value here
	} else {
		StatefulFeatureFunction::SetParameter(key, value);
	}
}

}

