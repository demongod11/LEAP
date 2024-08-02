#ifndef ABC__GetNCInfo_h
#define ABC__GetNCInfo_h

#include "base/main/main.h"

ABC_NAMESPACE_HEADER_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

/*=== getNCInfo.c =============================================================*/
extern int Abc_getNCInfo(Abc_Ntk_t * pNtk, char* nodesFile, char* cutsFile, double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose);

#endif
ABC_NAMESPACE_HEADER_END