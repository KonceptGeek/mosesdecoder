#pragma once

#include <string>
#include "StatefulFeatureFunction.h"
#include "FFState.h"

namespace Moses {

class CoarseBiLMState: public FFState {
	int m_targetLen;
public:
	CoarseBiLMState(int targetLen) :
			m_targetLen(targetLen) {
	}

	int Compare(const FFState& other) const;
};

class CoarseBiLM: public StatefulFeatureFunction {

protected:
	std::map<std::string, std::string> tgtWordToClusterId;
	std::map<std::string, std::string> srcWordToClusterId;
	std::map<std::string, std::string> bitokenToBitokenId;
	std::map<std::string, std::string> bitokenIdToClusterId;

public:
	CoarseBiLM(const std::string &line);

	bool IsUseable(const FactorMask &mask) const {
		return true;
	}
	virtual const FFState* EmptyHypothesisState(const InputType &input) const {
		return new CoarseBiLMState(0);
	}

	void EvaluateInIsolation(const Phrase &source,
			const TargetPhrase &targetPhrase,
			ScoreComponentCollection &scoreBreakdown,
			ScoreComponentCollection &estimatedFutureScore) const;
	void EvaluateWithSourceContext(const InputType &input,
			const InputPath &inputPath, const TargetPhrase &targetPhrase,
			const StackVec *stackVec, ScoreComponentCollection &scoreBreakdown,
			ScoreComponentCollection *estimatedFutureScore = NULL) const;

	void EvaluateTranslationOptionListWithSourceContext(const InputType &input,
			const TranslationOptionList &translationOptionList) const;

	FFState* EvaluateWhenApplied(const Hypothesis& cur_hypo,
			const FFState* prev_state,
			ScoreComponentCollection* accumulator) const;
	FFState* EvaluateWhenApplied(const ChartHypothesis& /* cur_hypo */,
			int /* featureID - used to index the state in the previous hypotheses */,
			ScoreComponentCollection* accumulator) const;

	void SetParameter(const std::string& key, const std::string& value);

	std::map<std::string, std::string> LoadManyToOneMap(const std::string& path);
};

}
