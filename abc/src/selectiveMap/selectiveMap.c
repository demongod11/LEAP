#include "base/abc/abc.h"
#include "base/main/main.h"
#include "base/main/mainInt.h"
#include "proof/fraig/fraig.h"
#include "opt/fxu/fxu.h"
#include "opt/fxch/Fxch.h"
#include "opt/cut/cut.h"
#include "map/fpga/fpga.h"
#include "map/if/if.h"
#include "opt/sim/sim.h"
#include "opt/res/res.h"
#include "opt/lpk/lpk.h"
#include "aig/gia/giaAig.h"
#include "opt/dar/dar.h"
#include "opt/mfs/mfs.h"
#include "proof/fra/fra.h"
#include "aig/saig/saig.h"
#include "proof/int/int.h"
#include "proof/dch/dch.h"
#include "proof/ssw/ssw.h"
#include "opt/cgt/cgt.h"
#include "bool/kit/kit.h"
#include "map/amap/amap.h"
#include "opt/ret/retInt.h"
#include "sat/xsat/xsat.h"
#include "sat/satoko/satoko.h"
#include "sat/cnf/cnf.h"
#include "proof/cec/cec.h"
#include "proof/acec/acec.h"
#include "proof/pdr/pdr.h"
#include "misc/tim/tim.h"
#include "bdd/llb/llb.h"
#include "bdd/bbr/bbr.h"
#include "map/cov/cov.h"
#include "base/cmd/cmd.h"
#include "proof/abs/abs.h"
#include "sat/bmc/bmc.h"
#include "proof/ssc/ssc.h"
#include "opt/sfm/sfm.h"
#include "opt/sbd/sbd.h"
#include "bool/rpo/rpo.h"
#include "map/mpm/mpm.h"
#include "opt/fret/fretime.h"
#include "opt/nwk/nwkMerge.h"
#include "base/acb/acbPar.h"
#include "map/mapper/mapperInt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map/mio/mioInt.h>
#include <map/mio/mio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

Abc_Ntk_t * Abc_selectiveMap ( Abc_Ntk_t * pNtk, char * cutChoiceFile, char* dumpFile , double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose);
const void* extractChoices(char* line, Map_Man_t * pMan);
const void* extractMapping(Map_Man_t * pMan, FILE* fp);

extern Vec_Int_t * Sim_NtkComputeSwitching( Abc_Ntk_t * pNtk, int nPatterns );
extern Map_Man_t *  Abc_NtkToMap( Abc_Ntk_t * pNtk, double DelayTarget, int fRecovery, float * pSwitching, int fVerbose );
extern Abc_Ntk_t *  Abc_NtkFromMap( Map_Man_t * pMan, Abc_Ntk_t * pNtk, int fUseBuffs );
extern int Map_Mapping( Map_Man_t * p );
extern int Map_MatchNodeCut( Map_Man_t * p, Map_Node_t * pNode, Map_Cut_t * pCut, int fPhase, float fWorstLimit );
////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////  

int comp (const void * elem1, const void * elem2) 
{
    int f = *((int*)elem1);
    int s = *((int*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

const void* extractChoices(char* line, Map_Man_t * pMan)
{
    char* tmp = strdup(line);
    const char* tok;
    tok = strtok(line, ",");
    int flag = 0, flag2 = 0;
    int nodeIdx = 0; 
    int cutCounter = 0;
    while(tok != NULL)
    {
        if (flag == 0) {
            nodeIdx = atoi(tok); 
            flag = 1;
        }
        else cutCounter++;
        tok = strtok(NULL, ",");
    }
    pMan->vMapObjs->pArray[nodeIdx]->numCutChoices = cutCounter;
    pMan->vMapObjs->pArray[nodeIdx]->cutChoices = malloc(cutCounter * sizeof(long long int));
    tok = strtok(tmp, ",");
    cutCounter = 0;
    while(tok != NULL)
    {
        if (flag2 == 0) flag2 = 1;
        else {
            pMan->vMapObjs->pArray[nodeIdx]->cutChoices[cutCounter] = atoi(tok);
            cutCounter++;
        }
        tok = strtok(NULL, ",");
    }
    qsort(pMan->vMapObjs->pArray[nodeIdx]->cutChoices, cutCounter, sizeof(long long int), comp);
    free(tmp);
    return;
}

const void* print_cutChoices(Map_Man_t* pMan){
    Map_Node_t* pNode;
    for(long long int i = 0; i < pMan->vMapObjs->nSize; i++){
        pNode = pMan->vMapObjs->pArray[i];
        Abc_Print(1, "%d,", pNode->Num);
        for(long long int j = 0; j < pNode->numCutChoices; j++){
            Abc_Print(1, "%lld,", pNode->cutChoices[j]);
        }
        Abc_Print(1, "\n");
    }
    return;
}

const void* countTotalCutChoices(Map_Man_t* pMan){
    Map_Node_t* pNode;
    long long int cnt = 0;
    for(long long int i = 0; i < pMan->vMapObjs->nSize; i++){
        pNode = pMan->vMapObjs->pArray[i];
        cnt+=(pNode->numCutChoices);
    }
    Abc_Print(1, "Total number of cut choices = %lld\n", cnt);
    return;
}

const void* free_cutChoices(Map_Man_t* pMan){
    Map_Node_t* pNode;
    for(long long int i = 0; i < pMan->vMapObjs->nSize; i++){
        pNode = pMan->vMapObjs->pArray[i];
        free(pNode->cutChoices);
    }
    return;
}

const void* extractMapping_rec(Map_Man_t* pMan, Map_Node_t* pNode, FILE* fp, int* vis){
    Map_Cut_t * pCut;
    Map_Node_t * pNodeR;

    // get the regular node and its phase
    pNodeR = Map_Regular(pNode);

    // quit if the node was already visited in this phase
    if(vis[pNodeR->Num] == 1) return;
    vis[pNodeR->Num] = 1;
    // quit if this is a PI node
    if ( Map_NodeIsVar(pNodeR) ){
        return;
    }
    // propagate through buffer
    if ( Map_NodeIsBuf(pNodeR) )
    {
        extractMapping_rec( pMan, pNodeR->p1, fp, vis );
        return;
    }
    assert( Map_NodeIsAnd(pNode) );
    // get the cut implementing this or opposite polarity
    pCut = pNodeR->pCutBest[0];
    if ( pCut == NULL )
    {
        pCut   = pNodeR->pCutBest[1];
    }

    for ( int i = 0; i < pCut->nLeaves; i++ )
    {
        if(pCut->ppLeaves[i]){
            Map_Cut_t * pChildCut;
            Map_Node_t * pChildNodeR;
            // get the regular node and its phase
            pChildNodeR = Map_Regular(pCut->ppLeaves[i]);

            // quit if this is a PI node
            if ( Map_NodeIsVar(pChildNodeR) ){
                return;
            }
            // propagate through buffer
            if ( Map_NodeIsBuf(pChildNodeR) )
            {
                extractMapping_rec( pMan, pChildNodeR->p1, fp, vis );
                return;
            }
            assert( Map_NodeIsAnd(pChildNodeR) );
            // get the cut implementing this or opposite polarity
            pChildCut = pChildNodeR->pCutBest[0];
            if ( pChildCut == NULL )
            {
                pChildCut   = pChildNodeR->pCutBest[1];
            }
            fprintf(fp, "%lld,%lld\n", pCut->cutIdx, pChildCut->cutIdx);
            extractMapping_rec( pMan, pCut->ppLeaves[i], fp, vis );
        }
    }
    return;
}

const void* extractMapping(Map_Man_t* pMan, FILE* fp){
    Map_Node_t * pNode;
    int total_nodes = pMan->vMapObjs->nSize;
    int vis[total_nodes];
    for(int i = 0; i < total_nodes; i++) vis[i] = -1;

    // visit nodes reachable from POs in the DFS order through the best cuts
    for ( int i = 0; i < pMan->nOutputs; i++ )
    {
        pNode = pMan->pOutputs[i];
        if ( !Map_NodeIsConst(pNode) )
            extractMapping_rec( pMan, pNode, fp, vis );
    }
    return;
}   

const void* extractCutStats(Map_Man_t* p, FILE* fp){
    Map_Node_t * pNode;
    Map_Cut_t * pCut;
    float cut_delay_0, cut_delay_1;
    for(long long int i = 0; i < p->vMapObjs->nSize; i++){
        pNode = p->vMapObjs->pArray[i];
        if(pNode->pCuts->pNext == NULL) continue;
        for ( pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext ){
            Map_MatchNodeCut( p, pNode, pCut, 0, MAP_FLOAT_LARGE );
            if(pCut->M[0].pSuperBest == NULL){
                cut_delay_0 = -1;
            } 
            else{
                cut_delay_0 = MAP_MAX(pCut->M[0].pSuperBest->tDelayMax.Rise, pCut->M[0].pSuperBest->tDelayMax.Fall);
            }
            fprintf(fp, "%lld,0,%u,%10.6f\n", pCut->cutIdx, pCut->uCanon[0][0], cut_delay_0);

            Map_MatchNodeCut( p, pNode, pCut, 1, MAP_FLOAT_LARGE );
            if(pCut->M[1].pSuperBest == NULL){
                cut_delay_1 = -1;
            } 
            else{
                cut_delay_1 = MAP_MAX(pCut->M[1].pSuperBest->tDelayMax.Rise, pCut->M[1].pSuperBest->tDelayMax.Fall);
            } 
            fprintf(fp, "%lld,1,%u,%10.6f\n", pCut->cutIdx, pCut->uCanon[1][0], cut_delay_1);
        }
    }
}

const void* printSuperStats(Map_Man_t* p, FILE* fp){
    Abc_Print(1, "Name of supergate library is %s\n", p->pSuperLib->pName);
    Abc_Print(1, "nSupersAll is %d\n", p->pSuperLib->nSupersAll);
    Abc_Print(1, "nSupersReal is %d\n", p->pSuperLib->nSupersReal);
    Abc_Print(1, "Name of the library is %s\n", p->pSuperLib->pGenlib->pName);
    Abc_Print(1, "The number of the gates is %d\n\n", p->pSuperLib->pGenlib->nGates);

    Map_Node_t * pNode;
    Map_Cut_t * pCut;
    Map_Super_t * pSuper;
    Map_Match_t* pMatch;
    long long int cnt = 0;
    for(long long int i = 0; i < p->vMapObjs->nSize; i++){
        pNode = p->vMapObjs->pArray[i];
        if(pNode->pCuts->pNext == NULL) continue;
        for ( pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext ){
            fprintf(fp, "%d,%lld,0,%u ---> ", pNode->Num, pCut->cutIdx, pCut->uTruth);
            pMatch = pCut->M + 0;
            cnt = 0;
            for ( pSuper = pMatch->pSupers; pSuper; pSuper = pSuper->pNext){
                cnt++;
                fprintf(fp, "%d, %d -- %u,%u, ", pSuper->Num, pSuper->nPhases, pSuper->uTruth[0], pSuper->uTruth[1]);
            }
            fprintf(fp, " ---> %lld\n", cnt);

            fprintf(fp, "%d,%lld,1,%u ---> ", pNode->Num, pCut->cutIdx, pCut->uTruth);
            pMatch = pCut->M + 1;
            cnt = 0;
            for ( pSuper = pMatch->pSupers; pSuper; pSuper = pSuper->pNext){
                cnt++;
                fprintf(fp, "%d, %d -- %u,%u, ", pSuper->Num, pSuper->nPhases, pSuper->uTruth[0], pSuper->uTruth[1]);
            }
            fprintf(fp, " ---> %lld\n", cnt);
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
    return;
}

Abc_Ntk_t * Abc_selectiveMap ( Abc_Ntk_t * pNtk, char * cutChoiceFile, char* dumpFile , double DelayTarget, double AreaMulti, double DelayMulti, float LogFan, float Slew, float Gain, int nGatesMin, int fRecovery, int fSwitching, int fSkipFanout, int fUseProfile, int fUseBuffs, int fVerbose){
    assert(pNtk != NULL); 
    FILE * fp; 
    fp = fopen(cutChoiceFile, "r"); 

    static int fUseMulti = 0;
    int fShowSwitching = 1;

    Abc_Ntk_t * pNtkNew;
    Map_Man_t * pMan;
    Vec_Int_t * vSwitching = NULL;
    float * pSwitching = NULL;
    Mio_Library_t * pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    abctime clkStart = Abc_Clock();
    assert( Abc_NtkIsStrash(pNtk) );
    // derive library from SCL
    // if the library is created here, it will be deleted when pSuperLib is deleted in Map_SuperLibFree()
    if ( Abc_FrameReadLibScl() && Abc_SclHasDelayInfo( Abc_FrameReadLibScl() ) )
    {
        if ( pLib && Mio_LibraryHasProfile(pLib) )
            pLib = Abc_SclDeriveGenlib( Abc_FrameReadLibScl(), pLib, Slew, Gain, nGatesMin, fVerbose );
        else
            pLib = Abc_SclDeriveGenlib( Abc_FrameReadLibScl(), NULL, Slew, Gain, nGatesMin, fVerbose );
        if ( Abc_FrameReadLibGen() )
        {
            Mio_LibraryTransferDelays( (Mio_Library_t *)Abc_FrameReadLibGen(), pLib );
            Mio_LibraryTransferProfile( pLib, (Mio_Library_t *)Abc_FrameReadLibGen() );
        }
        // remove supergate library
        Map_SuperLibFree( (Map_SuperLib_t *)Abc_FrameReadLibSuper() );
        Abc_FrameSetLibSuper( NULL );
    }
    // quit if there is no library
    if ( pLib == NULL )
    {
        printf( "The current library is not available.\n" );
        return 0;
    }
    if ( AreaMulti != 0.0 )
        fUseMulti = 1, printf( "The cell areas are multiplied by the factor: <num_fanins> ^ (%.2f).\n", AreaMulti );
    if ( DelayMulti != 0.0 )
        fUseMulti = 1, printf( "The cell delays are multiplied by the factor: <num_fanins> ^ (%.2f).\n", DelayMulti );

    // penalize large gates by increasing their area
    if ( AreaMulti != 0.0 )
        Mio_LibraryMultiArea( pLib, AreaMulti );
    if ( DelayMulti != 0.0 )
        Mio_LibraryMultiDelay( pLib, DelayMulti );

    // derive the supergate library
    if ( fUseMulti || Abc_FrameReadLibSuper() == NULL )
    {
        if ( fVerbose )
            printf( "Converting \"%s\" into supergate library \"%s\".\n", 
                Mio_LibraryReadName(pLib), Extra_FileNameGenericAppend(Mio_LibraryReadName(pLib), ".super") );
        // compute supergate library to be used for mapping
        if ( Mio_LibraryHasProfile(pLib) )
            printf( "Abc_NtkMap(): Genlib library has profile.\n" );
        Map_SuperLibDeriveFromGenlib( pLib, fVerbose );
    }

    // return the library to normal
    if ( AreaMulti != 0.0 )
        Mio_LibraryMultiArea( (Mio_Library_t *)Abc_FrameReadLibGen(), -AreaMulti );
    if ( DelayMulti != 0.0 )
        Mio_LibraryMultiDelay( (Mio_Library_t *)Abc_FrameReadLibGen(), -DelayMulti );

    // print a warning about choice nodes
    if ( fVerbose && Abc_NtkGetChoiceNum( pNtk ) )
        printf( "Performing mapping with choices.\n" );

    // compute switching activity
    fShowSwitching |= fSwitching;
    if ( fShowSwitching )
    {
        extern Vec_Int_t * Sim_NtkComputeSwitching( Abc_Ntk_t * pNtk, int nPatterns );
        vSwitching = Sim_NtkComputeSwitching( pNtk, 4096 );
        pSwitching = (float *)vSwitching->pArray;
    }


    pMan = Abc_NtkToMap( pNtk, DelayTarget, fRecovery, pSwitching, fVerbose );
    if ( pMan == NULL )
        return NULL;
    char* line = NULL;
    size_t line_length = 0;
    while (getline(&line, &line_length, fp) != -1)
    {
        char* tmp = strdup(line);
        extractChoices(tmp, pMan);
        free(tmp);
    }
    // print_cutChoices(pMan);
    if ( pSwitching ) Vec_IntFree( vSwitching );
    if ( pMan == NULL )
        return NULL;
    Map_ManSetSwitching( pMan, 0 );
    Map_ManSetSkipFanout( pMan, 0 );
    if ( fUseProfile )
        Map_ManSetUseProfile( pMan );
    if ( LogFan != 0 )
        Map_ManCreateNodeDelays( pMan, LogFan );
    if ( !Map_Mapping( pMan ) )
    {
        Map_ManFree( pMan );
        Abc_Print(1, "Mapping Failed\n");
        return NULL;
    }

    // // Extracting Cut Stats
    // FILE* cut_stats_fp = fopen("../data/multiplier/cut_stats.csv", "w+");
    // fprintf(cut_stats_fp,"cut_idx,fPhase,canon_tt,cut_delay\n");
    // extractCutStats(pMan,cut_stats_fp);
    // fclose(cut_stats_fp);

    // // Print Supergates stats
    // FILE* super_stats_fp = fopen("../data/adder/super_stats.txt", "w+");
    // printSuperStats(pMan, super_stats_fp);
    // fclose(super_stats_fp);

    // reconstruct the network after mapping (use buffers when user requested or in the area mode))
    pNtkNew = Abc_NtkFromMap( pMan, pNtk, fUseBuffs || (DelayTarget == (double)ABC_INFINITY) );
    if ( Mio_LibraryHasProfile(pLib) )
        Mio_LibraryTransferProfile2( (Mio_Library_t *)Abc_FrameReadLibGen(), pLib );

    // extract and dump the cuts used in the mapping
    FILE* dumpFile_fp = fopen(dumpFile, "w+");
    fprintf(dumpFile_fp,"cut1_idx,cut2_idx\n");
    extractMapping(pMan,dumpFile_fp);
    fclose(dumpFile_fp);

    free_cutChoices(pMan);
    Map_ManFree( pMan );
    if ( pNtkNew == NULL )
        return NULL;

    if ( pNtk->pExdc )
        pNtkNew->pExdc = Abc_NtkDup( pNtk->pExdc );

    // make sure that everything is okay
    if ( !Abc_NtkCheck( pNtkNew ) )
    {
        printf( "Abc_NtkMap: The network check has failed.\n" );
        Abc_NtkDelete( pNtkNew );
        return NULL;
    }
    if ( fVerbose )
    {
        ABC_PRT( "Total runtime", Abc_Clock() - clkStart );
    }
    return pNtkNew; 
}

ABC_NAMESPACE_IMPL_END