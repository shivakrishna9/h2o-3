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
      TaskEverything[] cvs = new TaskEverything[ncoly];

      double[] xmeans = new double[ncolx];
      for (int x = 0; x < ncoly; x++)
        xmeans[x] = vecxs[x].mean();

      // Launch tasks; each does all Xs vs one Y
      for (int y = 0; y < ncoly; y++) {
        cvs[y] = new TaskEverything(vecys[y].mean(), xmeans).dfork(new Frame(vecys[y]).add(frx));
      }


      //Calculate standard deviation of x and standard deviation of y
      double[] sigmax = new double[ncoly];
      double[] sigmay = new double[ncoly];
      double[] denom = new double[ncoly];

      // 1-col returns scalar
      if (ncolx == 1 && ncoly == 1) {
        sigmax[0] = Math.sqrt(Math.abs(cvs[0].getResult()._denomx[0] / (frx.numRows() - 1)));
        sigmay[0] = Math.sqrt(Math.abs(cvs[0].getResult()._denomy[0] / (fry.numRows() - 1)));
        denom[0] = sigmax[0]*sigmay[0];
        return new ValNum((cvs[0].getResult()._cor[0] / (fry.numRows() - 1))/denom[0]);
      }

      for (int y = 0; y < ncoly; y++) {
        //Calculate standard deviation of y
        sigmay[y] = Math.sqrt(Math.abs(cvs[y].getResult()._denomy[y] / (fry.numRows() - 1)));
        //Calculate standard deviation of x
        sigmax[y] = Math.sqrt(Math.abs(cvs[y].getResult()._denomx[y] / (frx.numRows() - 1)));
      }

      //Denominator for correlation calculation is sigma_x * sigma_y
      denom = ArrayUtils.mult(sigmax,sigmay);

      //Gather final result, which is the correlation coefficient per column
      Vec[] res = new Vec[ncoly];
      Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
      for (int y = 0; y < ncoly; y++) {
        res[y] = Vec.makeVec(ArrayUtils.div(ArrayUtils.div(cvs[y].getResult()._cor, (fry.numRows() - 1)), denom[y]), keys[y]);
      }

      return new ValFrame(new Frame(fry._names, res));
    } else { //if (mode.equals(Mode.CompleteObs))

      TaskCompleteObsMean taskCompleteObsMean = new TaskCompleteObsMean(ncoly, ncolx).doAll(new Frame(fry).add(frx));
      long NACount = taskCompleteObsMean._NACount;
      double[] ymeans = ArrayUtils.div(taskCompleteObsMean._ysum, (fry.numRows() - NACount));
      double[] xmeans = ArrayUtils.div(taskCompleteObsMean._xsum, (fry.numRows() - NACount));

      // 1 task with all Xs and Ys
      TaskCompleteObs cvs = new TaskCompleteObs(ymeans, xmeans).doAll(new Frame(fry).add(frx));

      //Calculate standard deviation of x and standard deviation of y
      double[] sigmax = new double[ncoly];
      double[] sigmay = new double[ncoly];
      double[] denom = new double[ncoly];

      // 1-col returns scalar
      // 1-col returns scalar
      if (ncolx == 1 && ncoly == 1) {
        sigmax[0] = Math.sqrt(Math.abs(cvs._denomx[0][0] / (frx.numRows() - 1 - NACount)));
        sigmay[0] = Math.sqrt(Math.abs(cvs._denomy[0][0] / (fry.numRows() - 1 - NACount)));
        denom[0] = sigmax[0]*sigmay[0];
        return new ValNum((cvs.getResult()._cor[0][0] / (fry.numRows() - 1 - NACount))/denom[0]);
      }

      for (int y = 0; y < ncoly; y++) {
        //Calculate standard deviation of y
        sigmay[y] = Math.sqrt(Math.abs(cvs._denomy[y][y] / (fry.numRows() - 1 - NACount)));
        //Calculate standard deviation of x
        sigmax[y] = Math.sqrt(Math.abs(cvs._denomx[y][y] / (frx.numRows() - 1 - NACount)));
      }

      //Denominator for correlation calculation is sigma_x * sigma_y
      denom = ArrayUtils.mult(sigmax,sigmay);

      // Gather all the Xs-vs-Y correlation arrays; divide by rows
      Vec[] res = new Vec[ncoly];
      Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
      for (int y = 0; y < ncoly; y++) {
        res[y] = Vec.makeVec(ArrayUtils.div(ArrayUtils.div(cvs._cor[y], (fry.numRows() - 1 - NACount)), denom[y]), keys[y]);
      }
      return new ValFrame(new Frame(fry._names, res));
    }
  }

  private static class TaskEverything extends MRTask<TaskEverything> {
    double[] _cor;
    double[] _denomx;
    double[] _denomy;
    final double _xmeans[], _ymean;

    TaskEverything(double ymean, double[] xmeans) {
      _ymean = ymean;
      _xmeans = xmeans;
    }

    @Override
    public void map(Chunk cs[]) {
      final int ncolsx = cs.length - 1;
      final Chunk cy = cs[0];
      final int len = cy._len;
      _cor = new double[ncolsx];
      _denomx = new double[ncolsx];
      _denomy = new double[ncolsx];
      double sum;
      double sumx;
      double sumy;
      for (int x = 0; x < ncolsx; x++) {
        sum = 0;
        sumx = 0;
        sumy = 0;
        final Chunk cx = cs[x + 1];
        final double xmean = _xmeans[x];
        for (int row = 0; row < len; row++) {
          sumx += (cx.atd(row) - xmean) * (cx.atd(row) - xmean);
          sumy += (cy.atd(row) - _ymean) * (cy.atd(row) - _ymean);
          sum += (cx.atd(row) - xmean) * (cy.atd(row) - _ymean);
        }
        _cor[x] = sum;
        _denomx[x] = sumx;
        _denomy[x] = sumy;
      }
    }

    @Override
    public void reduce(TaskEverything cvt) {
      ArrayUtils.add(_cor, cvt._cor);
      ArrayUtils.add(_denomx, cvt._denomx);
      ArrayUtils.add(_denomy, cvt._denomy);
    }
  }

  private static class TaskCompleteObsMean extends MRTask<TaskCompleteObsMean> {
    double[] _xsum, _ysum;
    long _NACount;
    int _ncolx, _ncoly;

    TaskCompleteObsMean(int ncoly, int ncolx) {
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
    public void reduce(TaskCompleteObsMean cvt) {
      ArrayUtils.add(_xsum, cvt._xsum);
      ArrayUtils.add(_ysum, cvt._ysum);
      _NACount += cvt._NACount;
    }
  }

  private static class TaskCompleteObs extends MRTask<TaskCompleteObs> {
    double[][] _cor;
    double[][] _denomx;
    double[][] _denomy;
    final double _xmeans[], _ymeans[];

    TaskCompleteObs(double[] ymeans, double[] xmeans) {
      _ymeans = ymeans;
      _xmeans = xmeans;
    }

    @Override
    public void map(Chunk cs[]) {
      int ncolx = _xmeans.length;
      int ncoly = _ymeans.length;
      double[] xvals = new double[ncolx];
      double[] yvals = new double[ncoly];
      _cor = new double[ncoly][ncolx];
      _denomx = new double[ncoly][ncolx];
      _denomy = new double[ncoly][ncolx];
      double[] _cor_y;
      double[] _denom_x;
      double[] _denom_y;
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
            _cor_y = _cor[y];
            _denom_x = _denomx[y];
            _denom_y = _denomy[y];
            yval = yvals[y];
            ymean = _ymeans[y];
            for (int x = 0; x < ncolx; x++) {
              _cor_y[x] += (xvals[x] - _xmeans[x]) * (yval - ymean);
              _denom_x[x] += (xvals[x] - _xmeans[x]) * (xvals[x] - _xmeans[x]);
              _denom_y[x] += (yval - ymean) * (yval - ymean);
            }
          }
        }
      }
    }

    @Override
    public void reduce(TaskCompleteObs cvt) {
      ArrayUtils.add(_cor, cvt._cor);
      ArrayUtils.add(_denomx, cvt._denomx);
      ArrayUtils.add(_denomy, cvt._denomy);
    }
  }

}
