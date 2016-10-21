package water.rapids.ast.prims.advmath;

import water.Key;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.rapids.Env;
import water.rapids.Val;
import water.rapids.vals.ValFrame;
import water.rapids.vals.ValNum;
import water.rapids.ast.AstPrimitive;
import water.rapids.ast.AstRoot;
import water.util.ArrayUtils;

import java.util.Arrays;

/**
 * Calculate Pearson's Correlation Coefficient between columns of a frame
 * <p/>
 * Formula:
 * Pearson's Correlation Coefficient = Cov(X,Y)/sigma(X) * sigma(Y)
 */
public class AstCorrelation extends AstPrimitive {
  @Override
  public String[] args() {
    return new String[]{"ary", "x", "y", "use"};
  }

  private enum Mode {Everything, AllObs, CompleteObs}

  @Override
  public int nargs() {
    return 1 + 3; /* (cor X Y use) */
  }

  @Override
  public String str() {
    return "cor";
  }

  @Override
  public Val apply(Env env, Env.StackHelp stk, AstRoot asts[]) {
    Frame frx = stk.track(asts[1].exec(env)).getFrame();
    Frame fry = stk.track(asts[2].exec(env)).getFrame();
    if (frx.numRows() != fry.numRows())
      throw new IllegalArgumentException("Frames must have the same number of rows, found " + frx.numRows() + " and " + fry.numRows());

    String use = stk.track(asts[3].exec(env)).getStr();
    Mode mode;
    switch (use) {
      case "everything":
        mode = Mode.Everything;
        break;
      case "all.obs":
        mode = Mode.AllObs;
        break;
      case "complete.obs":
        mode = Mode.CompleteObs;
        break;
      default:
        throw new IllegalArgumentException("unknown use mode: " + use);
    }

    return fry.numRows() == 1 ? scalar(frx, fry, mode) : array(frx, fry, mode);
  }

  // Scalar correlation for 1 row
  private ValNum scalar(Frame frx, Frame fry, Mode mode) {
    if (frx.numCols() != fry.numCols())
      throw new IllegalArgumentException("Single rows must have the same number of columns, found " + frx.numCols() + " and " + fry.numCols());
    Vec vecxs[] = frx.vecs();
    Vec vecys[] = fry.vecs();
    double xmean = 0, ymean = 0, ncols = frx.numCols(), NACount = 0, xval, yval, ss = 0;
    for (int r = 0; r < ncols; r++) {
      xval = vecxs[r].at(0);
      yval = vecys[r].at(0);
      if (Double.isNaN(xval) || Double.isNaN(yval))
        NACount++;
      else {
        xmean += xval;
        ymean += yval;
      }
    }
    xmean /= (ncols - NACount);
    ymean /= (ncols - NACount);

    if (NACount != 0) {
      if (mode.equals(Mode.AllObs)) throw new IllegalArgumentException("Mode is 'all.obs' but NAs are present");
      if (mode.equals(Mode.Everything)) return new ValNum(Double.NaN);
    }

    for (int r = 0; r < ncols; r++) {
      xval = vecxs[r].at(0);
      yval = vecys[r].at(0);
      if (!(Double.isNaN(xval) || Double.isNaN(yval)))
        ss += (vecxs[r].at(0) - xmean) * (vecys[r].at(0) - ymean);
    }
    return new ValNum(ss / (ncols - NACount - 1));
  }

  // Matrix correlation.  Compute correlation between all columns from each Frame
  // against each other.  Return a matrix of correlations which is frx.numCols
  // wide and fry.numCols tall.
  private Val array(Frame frx, Frame fry, Mode mode) {
    Vec[] vecxs = frx.vecs();
    int ncolx = vecxs.length;
    Vec[] vecys = fry.vecs();
    int ncoly = vecys.length;

    if (mode.equals(Mode.Everything) || mode.equals(Mode.AllObs)) {

      if (mode.equals(Mode.AllObs)) {
        for (Vec v : vecxs)
          if (v.naCnt() != 0)
            throw new IllegalArgumentException("Mode is 'all.obs' but NAs are present");
      }
      CoVarTaskEverything[] cvs = new CoVarTaskEverything[ncoly];

      double[] xmeans = new double[ncolx];
      for (int x = 0; x < ncoly; x++) {
        xmeans[x] = vecxs[x].mean();
      }

      //Set up double arrays to capture sd(y) and sd(x) * sd(y)
      double[] sigmay = new double[ncoly];
      double[] sigmax = new double[ncoly];
      double[] denom;

      // Launch tasks; each does all Xs vs one Y
      for (int y = 0; y < ncoly; y++) {
        //Get covariance between x and y
        cvs[y] = new CoVarTaskEverything(vecys[y].mean(), xmeans).dfork(new Frame(vecys[y]).add(frx));
        //Get sigma_x and sigma_y
        sigmax[y] = vecxs[y].sigma();
        sigmay[y] = vecys[y].sigma();
      }

      //Denominator for correlation calculation is sigma_x * sigma_y
      denom = ArrayUtils.mult(sigmax,sigmay);

      // 1-col returns scalar
      if (ncolx == 1 && ncoly == 1) {
        return new ValNum((cvs[0].getResult()._covs[0] / (fry.numRows() - 1))/denom[0]);
      }

      //Gather final result, which is the correlation coefficient per column
      Vec[] res = new Vec[ncoly];
      Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
      for (int y = 0; y < ncoly; y++) {
        res[y] = Vec.makeVec(ArrayUtils.div(ArrayUtils.div(cvs[y].getResult()._covs, (fry.numRows() - 1)), denom[y]), keys[y]);
      }

      return new ValFrame(new Frame(fry._names, res));
    } else { //if (mode.equals(Mode.CompleteObs))

      CoVarTaskCompleteObsMean taskCompleteObsMean = new CoVarTaskCompleteObsMean(ncoly, ncolx).doAll(new Frame(fry).add(frx));
      long NACount = taskCompleteObsMean._NACount;
      double[] ymeans = ArrayUtils.div(taskCompleteObsMean._ysum, (fry.numRows() - NACount));
      double[] xmeans = ArrayUtils.div(taskCompleteObsMean._xsum, (fry.numRows() - NACount));

      // 1 task with all Xs and Ys
      CoVarTaskCompleteObs cvs = new CoVarTaskCompleteObs(ymeans, xmeans).doAll(new Frame(fry).add(frx));

      //Set up double arrays to capture sd(y) and sd(x) * sd(y)
      double[] sigmay = new double[ncoly];
      double[] sigmax = new double[ncoly];
      double[] denom = new double[ncoly];

      // Launch tasks; each does all Xs vs one Y
      for (int y = 0; y < ncoly; y++) {
        //Get sigma_x and sigma_y
        sigmay[y] = vecys[y].sigma();
        sigmax[y] = vecxs[y].sigma();
      }

      //Denominator for correlation calculation is sigma_x * sigma_y
      denom = ArrayUtils.mult(sigmax,sigmay);

      // 1-col returns scalar
      if (ncolx == 1 && ncoly == 1) {
        return new ValNum((cvs._covs[0][0] / (fry.numRows() - 1 - NACount))/denom[0]);
      }

      //Gather final result, which is the correlation coefficient per column
      Vec[] res = new Vec[ncoly];
      Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
      for (int y = 0; y < ncoly; y++) {
        res[y] = Vec.makeVec(ArrayUtils.div(ArrayUtils.div(cvs._covs[y], (fry.numRows() - 1 - NACount)), denom[y]), keys[y]);
      }

      return new ValFrame(new Frame(fry._names, res));
    }
  }

  private static class CoVarTaskEverything extends MRTask<CoVarTaskEverything> {
    double[] _covs;
    final double _xmeans[], _ymean;

    CoVarTaskEverything(double ymean, double[] xmeans) {
      _ymean = ymean;
      _xmeans = xmeans;
    }

    @Override
    public void map(Chunk cs[]) {
      final int ncolsx = cs.length - 1;
      final Chunk cy = cs[0];
      final int len = cy._len;
      _covs = new double[ncolsx];
      double sum;
      for (int x = 0; x < ncolsx; x++) {
        sum = 0;
        final Chunk cx = cs[x + 1];
        final double xmean = _xmeans[x];
        for (int row = 0; row < len; row++)
          sum += (cx.atd(row) - xmean) * (cy.atd(row) - _ymean);
        _covs[x] = sum;
      }
    }

    @Override
    public void reduce(CoVarTaskEverything cvt) {
      ArrayUtils.add(_covs, cvt._covs);
    }
  }


  private static class CoVarTaskCompleteObsMean extends MRTask<CoVarTaskCompleteObsMean> {
    double[] _xsum, _ysum;
    long _NACount;
    int _ncolx, _ncoly;

    CoVarTaskCompleteObsMean(int ncoly, int ncolx) {
      _ncolx = ncolx;
      _ncoly = ncoly;
    }

    @Override
    public void map(Chunk cs[]) {
      _xsum = new double[_ncolx];
      _ysum = new double[_ncoly];

      double[] xvals = new double[_ncolx];
      double[] yvals = new double[_ncoly];

      double xval, yval;
      boolean add;
      int len = cs[0]._len;
      for (int row = 0; row < len; row++) {
        add = true;
        //reset existing arrays to 0 rather than initializing new ones to save on garbage collection
        Arrays.fill(xvals, 0);
        Arrays.fill(yvals, 0);

        for (int y = 0; y < _ncoly; y++) {
          final Chunk cy = cs[y];
          yval = cy.atd(row);
          //if any yval along a row is NA, discard the entire row
          if (Double.isNaN(yval)) {
            _NACount++;
            add = false;
            break;
          }
          yvals[y] = yval;
        }
        if (add) {
          for (int x = 0; x < _ncolx; x++) {
            final Chunk cx = cs[x + _ncoly];
            xval = cx.atd(row);
            //if any xval along a row is NA, discard the entire row
            if (Double.isNaN(xval)) {
              _NACount++;
              add = false;
              break;
            }
            xvals[x] = xval;
          }
        }
        //add is true iff row has been traversed and found no NAs among yvals and xvals
        if (add) {
          ArrayUtils.add(_xsum, xvals);
          ArrayUtils.add(_ysum, yvals);
        }
      }
    }

    @Override
    public void reduce(CoVarTaskCompleteObsMean cvt) {
      ArrayUtils.add(_xsum, cvt._xsum);
      ArrayUtils.add(_ysum, cvt._ysum);
      _NACount += cvt._NACount;
    }
  }

  private static class CoVarTaskCompleteObs extends MRTask<CoVarTaskCompleteObs> {
    double[][] _covs;
    final double _xmeans[], _ymeans[];

    CoVarTaskCompleteObs(double[] ymeans, double[] xmeans) {
      _ymeans = ymeans;
      _xmeans = xmeans;
    }

    @Override
    public void map(Chunk cs[]) {
      int ncolx = _xmeans.length;
      int ncoly = _ymeans.length;
      double[] xvals = new double[ncolx];
      double[] yvals = new double[ncoly];
      _covs = new double[ncoly][ncolx];
      double[] _covs_y;
      double xval, yval, ymean;
      boolean add;
      int len = cs[0]._len;
      for (int row = 0; row < len; row++) {
        add = true;
        //reset existing arrays to 0 rather than initializing new ones to save on garbage collection
        Arrays.fill(xvals, 0);
        Arrays.fill(yvals, 0);

        for (int y = 0; y < ncoly; y++) {
          final Chunk cy = cs[y];
          yval = cy.atd(row);
          //if any yval along a row is NA, discard the entire row
          if (Double.isNaN(yval)) {
            add = false;
            break;
          }
          yvals[y] = yval;
        }
        if (add) {
          for (int x = 0; x < ncolx; x++) {
            final Chunk cx = cs[x + ncoly];
            xval = cx.atd(row);
            //if any xval along a row is NA, discard the entire row
            if (Double.isNaN(xval)) {
              add = false;
              break;
            }
            xvals[x] = xval;
          }
        }
        //add is true iff row has been traversed and found no NAs among yvals and xvals
        if (add) {
          for (int y = 0; y < ncoly; y++) {
            _covs_y = _covs[y];
            yval = yvals[y];
            ymean = _ymeans[y];
            for (int x = 0; x < ncolx; x++)
              _covs_y[x] += (xvals[x] - _xmeans[x]) * (yval - ymean);
          }
        }
      }
    }

    @Override
    public void reduce(CoVarTaskCompleteObs cvt) {
      ArrayUtils.add(_covs, cvt._covs);
    }
  }

}
